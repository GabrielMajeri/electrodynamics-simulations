from math import pi
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import typer

from electrodynamics.beams import LaguerreGaussBeamParameters
from electrodynamics.constants import ELECTRON_CHARGE, ELECTRON_MASS, SPEED_OF_LIGHT
from electrodynamics.detector import (
    DetectorParameters,
    initialize_detector_positions_negative_z,
)
from electrodynamics.fields import compute_scattered_electric_and_magnetic_fields
from electrodynamics.initial_conditions import (
    generate_initial_particle_momenta_moving_towards_laser,
    generate_initial_positions_uniformly_on_disk,
    generate_initial_positions_uniformly_within_ball,
)
from electrodynamics.integrate import compute_next_momentum_rk4
from electrodynamics.jax import initialize_jax
from electrodynamics.plotting import (
    Arrow3D,
    plot_angular_momentum_distribution,
    plot_final_momentum_distribution,
)
from electrodynamics.polarization import Polarizations
from electrodynamics.pulse import PulseWithFlatPeakParameters

c = SPEED_OF_LIGHT
m_e = ELECTRON_MASS
q = ELECTRON_CHARGE

type PulseParameters = PulseWithFlatPeakParameters

type IterationState = tuple[
    float, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]


@jax.shard_map(
    in_specs=(P("i"),) * 2 + (None,) * 11,
    out_specs=(P("i"),) * 6,
)
def integrate_particles(
    initial_positions: jax.Array,
    initial_momenta: jax.Array,
    start_time: jdc.Static[float],
    end_time: jdc.Static[float],
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseParameters],
    central_frequency: jdc.Static[float],
    frequency_width: jdc.Static[float],
    num_frequencies: jdc.Static[int],
    spectrum_measurement_position: jdc.Static[jax.Array],
    detector_parameters: jdc.Static[DetectorParameters],
    detector_positions: jdc.Static[jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute the trajectory and the scattered radiation from particles
    under the action of the laser pulse.
    """

    # TODO: make this configurable
    central_frequencies = jnp.linspace(
        central_frequency - 0.1 * central_frequency / 2,
        central_frequency + 0.1 * central_frequency / 2,
        1,
    )

    frequencies = jnp.linspace(
        central_frequency - frequency_width / 2,
        central_frequency + frequency_width / 2,
        num_frequencies,
    )

    batch_size = len(initial_positions)

    scattered_electric_field = jnp.zeros(
        shape=(batch_size, num_frequencies, 3), dtype=jnp.complex128
    )
    scattered_magnetic_field = jnp.zeros(
        shape=(batch_size, num_frequencies, 3), dtype=jnp.complex128
    )

    scattered_electric_field = jax.lax.pcast(
        scattered_electric_field, "i", to="varying"
    )
    scattered_magnetic_field = jax.lax.pcast(
        scattered_magnetic_field, "i", to="varying"
    )

    num_detector_points = len(detector_positions)

    detector_electric_field = jnp.zeros(
        shape=(batch_size, num_detector_points, 3), dtype=jnp.complex128
    )
    detector_magnetic_field = jnp.zeros(
        shape=(batch_size, num_detector_points, 3), dtype=jnp.complex128
    )

    detector_electric_field = jax.lax.pcast(detector_electric_field, "i", to="varying")
    detector_magnetic_field = jax.lax.pcast(detector_magnetic_field, "i", to="varying")

    compute_scattered_fields_for_all_frequencies = jax.vmap(
        compute_scattered_electric_and_magnetic_fields,
        in_axes=(0, None, None, None, None),
        out_axes=1,
    )

    compute_scattered_fields_for_all_detector_positions = jax.vmap(
        compute_scattered_fields_for_all_frequencies,
        in_axes=(None, None, None, None, 0),
        out_axes=1,
    )

    num_time_steps = int((end_time - start_time) / time_step) + 1

    def scan_fn(u: IterationState, _: None) -> tuple[IterationState, None]:
        (
            proper_time,
            previous_positions,
            previous_momenta,
            previous_scattered_electric_field,
            previous_scattered_magnetic_field,
            previous_detector_electric_field,
            previous_detector_magnetic_field,
        ) = u

        new_momenta = compute_next_momentum_rk4(
            previous_positions,
            previous_momenta,
            time_step,
            laser_parameters,
            pulse_parameters,
        )
        new_position = previous_positions + time_step * new_momenta

        electric_field, magnetic_field = compute_scattered_fields_for_all_frequencies(
            frequencies,
            new_position,
            new_momenta,
            initial_positions,
            spectrum_measurement_position,
        )
        new_scattered_electric_field = (
            previous_scattered_electric_field + time_step * electric_field
        )
        new_scattered_magnetic_field = (
            previous_scattered_magnetic_field + time_step * magnetic_field
        )

        electric_field, magnetic_field = (
            compute_scattered_fields_for_all_detector_positions(
                central_frequencies,
                new_position,
                new_momenta,
                initial_positions,
                detector_positions,
            )
        )

        electric_field = electric_field.sum(axis=-2)
        magnetic_field = magnetic_field.sum(axis=-2)

        new_detector_electric_field = (
            previous_detector_electric_field + time_step * electric_field
        )
        new_detector_magnetic_field = (
            previous_detector_magnetic_field + time_step * magnetic_field
        )

        u_next = (
            proper_time + time_step,
            new_position,
            new_momenta,
            new_scattered_electric_field,
            new_scattered_magnetic_field,
            new_detector_electric_field,
            new_detector_magnetic_field,
        )
        return u_next, None

    (
        (
            _final_time,
            final_position,
            final_momentum,
            scattered_electric_field,
            scattered_magnetic_field,
            detector_electric_field,
            detector_magnetic_field,
        ),
        _,
    ) = jax.lax.scan(
        scan_fn,
        (
            start_time,
            initial_positions,
            initial_momenta,
            scattered_electric_field,
            scattered_magnetic_field,
            detector_electric_field,
            detector_magnetic_field,
        ),
        None,
        length=num_time_steps,
    )

    return (
        final_position,
        final_momentum,
        scattered_electric_field,
        scattered_magnetic_field,
        detector_electric_field,
        detector_magnetic_field,
    )


@jdc.pytree_dataclass
class IntegrationResult:
    timestamps: np.ndarray
    frequencies: np.ndarray

    final_positions: jax.Array
    final_momenta: jax.Array

    scattered_electric_field_spectrum: jax.Array
    scattered_magnetic_field_spectrum: jax.Array

    detector_electric_field: jax.Array
    detector_magnetic_field: jax.Array


def simulate_trajectories(
    initial_positions: jax.Array,
    initial_momenta: jax.Array,
    start_time: jdc.Static[float],
    end_time: jdc.Static[float],
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseParameters],
    central_frequency: jdc.Static[float],
    frequency_width: jdc.Static[float],
    num_frequencies: jdc.Static[int],
    spectrum_measurement_position: jax.Array,
    detector_parameters: jdc.Static[DetectorParameters],
    detector_positions: jdc.Static[jax.Array],
) -> IntegrationResult:
    print("Integrating trajectory for all particles and computing scattered field")

    num_devices = min(len(initial_positions), len(jax.devices()))
    # print(f"Mesh grid size: {num_devices}")

    mesh = jax.make_mesh((num_devices,), ("i",))
    with jax.set_mesh(mesh):
        batch_size = 1024
        num_batches = max(1, len(initial_positions) // batch_size)

        batches = zip(
            jnp.split(initial_positions, num_batches, axis=0),
            jnp.split(initial_momenta, num_batches, axis=0),
        )

        final_positions = []
        final_momenta = []
        scattered_electric_field_spectrum = jnp.zeros((num_frequencies, 3))
        scattered_magnetic_field_spectrum = jnp.zeros((num_frequencies, 3))
        detector_electric_field = jnp.zeros((len(detector_positions), 3))
        detector_magnetic_field = jnp.zeros((len(detector_positions), 3))

        for index, batch in enumerate(batches):
            print(f"Batch #{index}")
            batch_positions = jax.device_put(batch[0], P("i", None))
            batch_momenta = jax.device_put(batch[1], P("i", None))

            (
                batch_final_positions,
                batch_final_momenta,
                batch_scattered_electric_field_spectrum,
                batch_scattered_magnetic_field_spectrum,
                batch_detector_electric_field,
                batch_detector_magnetic_field,
            ) = integrate_particles(
                batch_positions,
                batch_momenta,
                start_time,
                end_time,
                time_step,
                laser_parameters,
                pulse_parameters,
                central_frequency,
                frequency_width,
                num_frequencies,
                spectrum_measurement_position,
                detector_parameters,
                detector_positions,
            )

            final_positions.append(batch_final_positions.block_until_ready())
            final_momenta.append(batch_final_momenta.block_until_ready())
            scattered_electric_field_spectrum += jnp.sum(
                batch_scattered_electric_field_spectrum, axis=0
            ).block_until_ready()
            scattered_magnetic_field_spectrum += jnp.sum(
                batch_scattered_magnetic_field_spectrum, axis=0
            ).block_until_ready()
            detector_electric_field += jnp.sum(
                batch_detector_electric_field, axis=0
            ).block_until_ready()
            detector_magnetic_field += jnp.sum(
                batch_detector_magnetic_field, axis=0
            ).block_until_ready()

        final_positions = jnp.concatenate(final_positions)
        final_momenta = jnp.concatenate(final_momenta)

    return IntegrationResult(
        np.arange(start_time, end_time + 1e-10, time_step),
        np.linspace(
            central_frequency - frequency_width / 2,
            central_frequency + frequency_width / 2,
            num_frequencies,
        ),
        final_positions,
        final_momenta,
        scattered_electric_field_spectrum,
        scattered_magnetic_field_spectrum,
        detector_electric_field,
        detector_magnetic_field,
    )


@jdc.jit
def compute_angular_momentum(
    positions: jax.Array, momenta: jax.Array, particle_mass: jdc.Static[float]
) -> jax.Array:
    return particle_mass * (
        positions[:, 1] * momenta[:, 2] - positions[:, 2] * momenta[:, 1]
    )


def main() -> None:
    """Simulates a Laguerre-Gauss laser pulse hitting a bunch of electrons moving at relativistic speeds.
    The scattered radiation, as measuerd on a simulated detector, is Doppler-shifted to much higher frequencies.
    """

    initialize_jax()

    # Monochromatic laser
    laser_frequency = 0.057
    # ~800 nm, red light
    laser_wavelength = (2 * pi * c) / laser_frequency

    a_0 = 1e-1

    amplitude = a_0 * m_e * c * laser_frequency / abs(q)
    polarization = Polarizations.RIGHT_CIRCULAR.value
    waist_radius = 25 * laser_wavelength

    radial_index = 2
    azimuthal_index = -2

    laser_parameters = LaguerreGaussBeamParameters(
        laser_frequency,
        laser_wavelength,
        amplitude,
        polarization,
        waist_radius,
        radial_index,
        azimuthal_index,
    )

    seed = 42
    generator = np.random.default_rng(seed)
    num_electrons = 4096

    print(f"Working with {num_electrons} electrons")

    disk_radius = (1.75 + radial_index) * waist_radius

    initial_positions = generate_initial_positions_uniformly_on_disk(
        generator, disk_radius, num_electrons
    )

    # ball_radius = disk_radius
    # initial_positions = generate_initial_positions_uniformly_within_ball(
    #     generator, ball_radius, num_electrons
    # )

    # Add a 0 on the first index to obtain position 4-vectors
    initial_positions = np.concatenate(
        (np.zeros((num_electrons, 1), dtype=np.float64), initial_positions), axis=-1
    )

    # Relativistic factor
    gamma: float = 3

    initial_momenta = generate_initial_particle_momenta_moving_towards_laser(
        num_particles=num_electrons, gamma=gamma, particle_mass=m_e
    )

    initial_positions = jnp.asarray(initial_positions)
    initial_momenta = jnp.asarray(initial_momenta)

    ### Pulse parameters ###
    tau_0 = 10 / laser_frequency
    phi_0 = 3 * tau_0
    peak_duration_periods = 5

    pulse_parameters = PulseWithFlatPeakParameters(phi_0, tau_0, peak_duration_periods)

    integration_start_time = 0.0
    stretching_factor = gamma + np.sqrt(gamma**2 - 1)
    integration_end_time = (
        6 * tau_0 + peak_duration_periods * tau_0
    ) / stretching_factor
    time_step = ((2 * pi) / laser_frequency) / 100 / stretching_factor

    print(
        f"Integrating from t = {integration_start_time} to t = {integration_end_time}, with a time step of dt = {time_step}"
    )

    doppler_shift_factor = (
        (gamma + np.sqrt(gamma**2 - 1)) / (gamma - np.sqrt(gamma**2 - 1))
    ).item()

    central_frequency = laser_parameters.frequency * doppler_shift_factor
    print(f"Looking around Fourier frequency omega = {central_frequency}")

    frequency_width = 0.5 * central_frequency
    num_frequencies = 128

    detector_parameters = DetectorParameters(
        width=20 * 75 * laser_wavelength,
        height=20 * 75 * laser_wavelength,
        grid_size_x=64,
        grid_size_y=64,
    )
    z_distance = -10 * 100_000 * laser_wavelength

    detector_positions = initialize_detector_positions_negative_z(
        detector_parameters, z_distance
    )
    detector_positions = jnp.asarray(detector_positions)

    spectrum_measurement_position = jnp.array(
        (
            25 * laser_parameters.wavelength,
            -25 * laser_parameters.wavelength,
            z_distance,
        )
    )

    print("Plotting experimental setup")
    plots_directory = Path("plots")
    plots_directory.mkdir(parents=True, exist_ok=True)

    plot_setup(
        plots_directory,
        initial_positions,
        initial_momenta,
        detector_parameters,
        detector_positions,
    )

    print("Starting simulation")
    start_time = perf_counter()

    result = simulate_trajectories(
        initial_positions,
        initial_momenta,
        integration_start_time,
        integration_end_time,
        time_step,
        laser_parameters,
        pulse_parameters,
        central_frequency,
        frequency_width,
        num_frequencies,
        spectrum_measurement_position,
        detector_parameters,
        detector_positions,
    )

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Took {duration} seconds")

    print("Plotting results")

    ### Sample trajectory
    ### Final z-momentum distribution
    fig = plt.figure(dpi=200)

    plot_final_momentum_distribution(
        fig,
        np.asarray(initial_positions[:, 1:4]),
        waist_radius,
        np.asarray(result.final_momenta[:, 4]),
    )

    fig.savefig(plots_directory / "final_momenta.png")

    ### Final angular momentum distribution
    fig = plt.figure(dpi=200)

    angular_momenta = compute_angular_momentum(
        result.final_positions, result.final_momenta, m_e
    )

    plot_angular_momentum_distribution(
        fig,
        np.asarray(initial_positions[:, 1:4]),
        waist_radius,
        np.asarray(angular_momenta),
    )

    fig.savefig(plots_directory / "final_angular_momenta.png")

    ### Scattered electric field spectrum
    plot_scattered_radiation_spectrum(
        plots_directory,
        central_frequency,
        result.frequencies,
        np.asarray(result.scattered_electric_field_spectrum),
    )

    ### Electric/magnetic fields on detector screen
    detector_extent: tuple[float, float, float, float] = (
        (-detector_parameters.width / 2) / waist_radius,
        (+detector_parameters.width / 2) / waist_radius,
        (-detector_parameters.height / 2) / waist_radius,
        (+detector_parameters.height / 2) / waist_radius,
    )

    detector_electric_field = result.detector_electric_field.reshape(
        detector_parameters.grid_size_y, detector_parameters.grid_size_x, -1
    )

    for index, coordinate in enumerate(("x", "y", "z")):
        fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

        fig.suptitle("Scattered electric field on detector")

        ax.set_title(f"$E_{coordinate}$")

        image = ax.imshow(
            np.real(detector_electric_field[:, :, index]),
            cmap="bwr",
            extent=detector_extent,
        )

        # Uncomment to see where the frequency spectrum gets computed:
        # ax.scatter(
        #     spectrum_measurement_position[0] / waist_radius,
        #     spectrum_measurement_position[1] / waist_radius,
        #     marker="x",
        #     c="green",
        #     s=50,
        # )

        ax.set_xlabel("Detector $x/w_0$")
        ax.set_ylabel("Detector $y/w_0$")
        ax.grid()

        fig.colorbar(image)

        fig.tight_layout()
        fig.savefig(plots_directory / f"electric_field_{coordinate}.pdf")

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

    fig.suptitle("Scattered electric field on detector")

    ax.set_title("$|E|$")

    image = ax.imshow(
        np.linalg.vector_norm(detector_electric_field[:, :, :], axis=-1),
        cmap="bwr",
        extent=detector_extent,
    )

    ax.set_xlabel("Detector $x$")
    ax.set_ylabel("Detector $y$")
    ax.grid()

    fig.colorbar(image)

    fig.tight_layout()
    fig.savefig(plots_directory / "electric_field_norm.pdf")


def plot_setup(
    plots_directory: Path,
    positions: np.ndarray | jax.Array,
    momenta: np.ndarray | jax.Array,
    detector_parameters: DetectorParameters,
    detector_positions: np.ndarray | jax.Array,
) -> None:
    assert len(positions) == len(momenta), (
        "Positions and momenta arrays should correspond to the same number of particles"
    )
    num_particles = len(positions)
    if num_particles > 1024:
        # Sample a subset of the initial particle bunch
        sample_indices = np.arange(1024)
        positions = positions[sample_indices]
        momenta = momenta[sample_indices]

    assert positions.shape[-1] in (3, 4), (
        "Positions should be an array of 3-vectors or 4-vectors"
    )
    if positions.shape[-1] == 4:
        positions = positions[:, 1:4]

    assert momenta.shape[-1] in (3, 4), (
        "Momenta should be an array of 3-vectors or 4-vectors"
    )
    if momenta.shape[-1] == 4:
        momenta = momenta[:, 1:4]

    fig = plt.figure(figsize=(8, 5), dpi=300)
    fig.suptitle("Experimental setup")

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_title("Initial positions and momenta")

    # Draw the electrons
    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], s=3)  # pyright: ignore[reportArgumentType]

    # Draw initial momenta as arrows
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        momenta[:, 0],
        momenta[:, 1],
        momenta[:, 2],
        linewidth=1,
        length=5,
        normalize=True,
        color="orange",
    )

    ax.set_zlim(-10, 10)

    ax = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    ax.set_title("Detector, laser beam and particles")

    # Draw electrons again
    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], s=3)  # pyright: ignore[reportArgumentType]

    # Draw the detector mesh
    detector_xs = detector_positions[:, 0].reshape(
        detector_parameters.grid_size_y, detector_parameters.grid_size_x
    )
    detector_ys = detector_positions[:, 1].reshape(
        detector_parameters.grid_size_y, detector_parameters.grid_size_x
    )
    detector_zs = detector_positions[:, 2].reshape(
        detector_parameters.grid_size_y, detector_parameters.grid_size_x
    )

    # Plot as a surface mesh
    ax.plot_surface(
        detector_xs,
        detector_ys,
        detector_zs,
        edgecolor="royalblue",
        lw=0.5,
        alpha=0.3,
        zorder=1,
    )

    # Plot laser direction as an arrow
    detector_z = detector_positions[0, 2]
    laser_beam_direction_arrow = Arrow3D(
        (0, 0),
        (0, 0),
        (detector_z * 0.75, detector_z * 0.25),
        color="red",
        lw=3,
        arrowstyle="-|>",
        mutation_scale=10,
        zorder=5,
    )
    ax.add_artist(laser_beam_direction_arrow)

    # fig.tight_layout()
    fig.savefig(plots_directory / "experimental_setup.png")


def plot_scattered_radiation_spectrum(
    plots_directory: Path,
    central_frequency: float,
    frequencies: np.ndarray,
    scattered_electric_field_spectrum: np.ndarray,
) -> None:
    fig, ax = plt.subplots()

    ax.plot(
        frequencies,
        np.asarray(jnp.linalg.vector_norm(scattered_electric_field_spectrum, axis=-1)),
    )

    ax.axvline(central_frequency, linestyle="--", color="orange")

    ax.set_xlabel("Frequency $\\omega$")
    ax.set_ylabel("Amplitude")

    ax.set_title("$|E(\\omega)|$")
    ax.grid()

    fig.tight_layout()
    fig.savefig(plots_directory / "electric_field_spectrum.pdf")


if __name__ == "__main__":
    typer.run(main)
