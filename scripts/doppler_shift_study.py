from math import pi
import os
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import typer

from electrodynamics.beams import (
    LaguerreGaussBeamParameters,
    compute_laguerre_gauss_beam_fields,
)
from electrodynamics.constants import SPEED_OF_LIGHT
from electrodynamics.detector import DetectorParameters, initialize_detector_positions
from electrodynamics.initial_conditions import generate_initial_positions_on_disk
from electrodynamics.plotting import (
    plot_angular_momentum_distribution,
    plot_final_momentum_distribution,
)
from electrodynamics.polarization import Polarizations

c = SPEED_OF_LIGHT


@jdc.jit
def gaussian_envelope(
    phi: jax.Array, phi_0: jdc.Static[float], tau_0: jdc.Static[float]
) -> jax.Array:
    t = (phi - phi_0) / tau_0
    return jnp.exp(-(t * t))


@jdc.pytree_dataclass
class GaussianPulseParameters:
    phi_0: float
    tau_0: float


@jdc.jit
def compute_acceleration(
    momentum: jax.Array,
    electric_field: jax.Array,
    magnetic_field: jax.Array,
    charge_to_mass_ratio: jdc.Static[float],
) -> jax.Array:
    "Computes the acceleration felt by a charged particle in an electric field."

    u0, u1, u2, u3 = momentum.T
    E_x, E_y, E_z = electric_field.T
    B_x, B_y, B_z = magnetic_field.T

    du0 = u1 * E_x / c + u2 * E_y / c + u3 * E_z / c
    du1 = u0 * E_x / c + u2 * B_z - u3 * B_y
    du2 = u0 * E_y / c - u1 * B_z + u3 * B_x
    du3 = u0 * E_z / c + u1 * B_y - u2 * B_x

    return charge_to_mass_ratio * jnp.array((du0, du1, du2, du3)).T


@jdc.jit
def compute_intermediate_acceleration(
    time: float,
    position: jax.Array,
    momentum: jax.Array,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
) -> jax.Array:
    _, _, _, z = position.T

    modulation = gaussian_envelope(
        time - z / c, pulse_parameters.phi_0, pulse_parameters.tau_0
    )
    modulation = jnp.expand_dims(modulation, axis=-1)

    # electric_field, magnetic_field = compute_plane_wave_fields(
    #     laser_parameters, position
    # )
    electric_field, magnetic_field = compute_laguerre_gauss_beam_fields(
        laser_parameters, position
    )

    electric_field = modulation * electric_field
    magnetic_field = modulation * magnetic_field

    return compute_acceleration(
        momentum, electric_field, magnetic_field, charge_to_mass_ratio=-1
    )


@jdc.jit
def compute_new_momentum(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
) -> jax.Array:
    tc, _, _, _ = previous_position.T
    laboratory_time = tc / c

    # Euler
    # electric_field, magnetic_field = plane_wave_fields(
    #     laser_parameters, previous_position
    # )
    # acceleration = compute_acceleration(
    #     previous_momentum,
    #     electric_field,
    #     magnetic_field,
    #     charge_to_mass_ratio=-1,
    # )

    # Runge-Kutta 4th order
    k_1 = compute_intermediate_acceleration(
        laboratory_time,
        previous_position,
        previous_momentum,
        laser_parameters,
        pulse_parameters,
    )

    k_2 = compute_intermediate_acceleration(
        laboratory_time + time_step / 2,
        previous_position,
        previous_momentum + time_step / 2 * k_1,
        laser_parameters,
        pulse_parameters,
    )

    k_3 = compute_intermediate_acceleration(
        laboratory_time + time_step / 2,
        previous_position,
        previous_momentum + time_step / 2 * k_2,
        laser_parameters,
        pulse_parameters,
    )

    k_4 = compute_intermediate_acceleration(
        laboratory_time + time_step,
        previous_position,
        previous_momentum + time_step * k_3,
        laser_parameters,
        pulse_parameters,
    )

    acceleration = (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    # TODO: implement sanity checks
    # if (check_for_errors)
    # {
    #     check_integration_results(previous_momentum, acceleration);
    # }

    return previous_momentum + time_step * acceleration


def compute_scattered_fields(
    current_time: float,
    frequency: float | np.ndarray,
    position: jax.Array,
    momentum: jax.Array,
    initial_position: jax.Array,
    detector_position: np.ndarray | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Computes the scattered electric/magnetic field from a moving charge,
    using the formulas for the Liénard-Wiechert potentials.
    """
    particle_position = position[:, 1:4]
    particle_velocity = momentum[:, 1:4]
    initial_position = initial_position[:, 1:4]

    # r_0(t) = r(t) - R_0
    particle_displacement = particle_position - initial_position

    # x_0(t) = x - R_0
    detector_displacement = detector_position - initial_position

    # R(x_0, t) = x_0 - r_0(t) = (x - R_0) - (r(t) - R_0) = x - r(t)
    displacement = detector_displacement - particle_displacement
    displacement_norm = jnp.expand_dims(
        jnp.linalg.vector_norm(displacement, axis=-1), axis=-1
    )

    # n(x_0, t) = R(x_0, t)/|R(x_0, t)|
    view_direction = displacement / displacement_norm

    # exp(i * omega * (t + R(x_0, t)/c))
    oscillatory_kernel = jnp.exp(
        1j * frequency * (current_time + displacement_norm / c)
    )

    # \beta = v/c
    beta = particle_velocity / c

    # ===== Electric field terms =====
    # Common term: n(x_0, t) \times (n(x_0, t) \times \beta(t))
    electric_field_common_term = jnp.cross(
        view_direction, jnp.cross(view_direction, beta)
    )

    # O(1/|R|) term
    # - ((i * omega) / c) * (common term) / |R(x_0, t)|
    electric_field_first_term = -((1j * frequency) / c) * (
        electric_field_common_term / displacement_norm
    )

    displacement_norm_squared = displacement_norm * displacement_norm

    # O(1/|R|^2) term
    # [(common term) + n(x_0, t) * (1 + dot(n(x_0, t), \beta(t)))] / |R(x_0, t)|^2
    electric_field_second_term = (
        electric_field_common_term
        + view_direction
        * jnp.expand_dims(1 + jnp.linalg.vecdot(view_direction, beta), axis=-1)
    ) / displacement_norm_squared

    n_cross_beta = jnp.cross(view_direction, beta)

    # ===== Magnetic field terms =====
    # O(1/|R|) term
    magnetic_field_first_term = ((1j * frequency) / c) * (
        n_cross_beta / displacement_norm
    )

    # O(1/|R|^2) term
    magnetic_field_second_term = n_cross_beta / displacement_norm_squared

    # Add up the components
    electric_field = oscillatory_kernel * (
        electric_field_first_term + electric_field_second_term
    )
    magnetic_field = oscillatory_kernel * (
        magnetic_field_first_term + magnetic_field_second_term
    )

    return electric_field, magnetic_field


@jdc.jit
def compute_particle_trajectory(
    initial_position: jax.Array,
    initial_momentum: jax.Array,
    start_time: jdc.Static[float],
    end_time: jdc.Static[float],
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
) -> tuple[jax.Array, jax.Array]:
    "Compute the trajectory of a single particle under the action of the laser pulse."

    num_time_steps = int((end_time - start_time) / time_step) + 1

    def scan_fn(
        u: tuple[jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        previous_position, previous_momentum = u

        new_momentum = compute_new_momentum(
            previous_position,
            previous_momentum,
            time_step,
            laser_parameters,
            pulse_parameters,
        )
        new_position = previous_position + time_step * new_momentum

        u_next = (jnp.squeeze(new_position), jnp.squeeze(new_momentum))
        return u_next, u_next

    _, trajectory = jax.lax.scan(
        scan_fn, (initial_position, initial_momentum), None, length=num_time_steps
    )

    positions, momenta = trajectory
    return positions, momenta


type IterationState = tuple[
    float, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]


@jax.shard_map(
    in_specs=(P("i"),) * 2 + (None,) * 10,
    out_specs=(P("i"),) * 6,
)
def integrate_particles(
    initial_positions: jax.Array,
    initial_momenta: jax.Array,
    start_time: jdc.Static[float],
    end_time: jdc.Static[float],
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    central_frequency: jdc.Static[float],
    frequency_width: jdc.Static[float],
    num_frequencies: jdc.Static[int],
    detector_parameters: jdc.Static[DetectorParameters],
    detector_positions: jdc.Static[jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute the trajectory and the scattered radiation from particles
    under the action of the laser pulse.
    """

    frequencies = np.linspace(
        central_frequency - frequency_width / 2,
        central_frequency + frequency_width / 2,
        num_frequencies,
    )

    spectrum_measurement_position = np.array(
        (5 * laser_parameters.wavelength, 0.0, detector_parameters.z_distance)
        # (0 * laser_parameters.wavelength, 0.0, detector_parameters.z_distance)
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
        compute_scattered_fields,
        in_axes=(None, 0, None, None, None, None),
        out_axes=1,
    )

    compute_scattered_fields_for_all_detector_positions = jax.vmap(
        compute_scattered_fields,
        in_axes=(None, None, None, None, None, 0),
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

        new_momenta = compute_new_momentum(
            previous_positions,
            previous_momenta,
            time_step,
            laser_parameters,
            pulse_parameters,
        )
        new_position = previous_positions + time_step * new_momenta

        electric_field, magnetic_field = compute_scattered_fields_for_all_frequencies(
            proper_time,
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
                proper_time,
                central_frequency,
                new_position,
                new_momenta,
                initial_positions,
                detector_positions,
            )
        )
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

    electron_positions: jax.Array
    electron_momenta: jax.Array

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
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    central_frequency: jdc.Static[float],
    frequency_width: jdc.Static[float],
    num_frequencies: jdc.Static[int],
    detector_parameters: jdc.Static[DetectorParameters],
    detector_positions: jdc.Static[jax.Array],
) -> IntegrationResult:
    print("Integrating trajectory for sample particle")

    electron_positions, electron_momenta = compute_particle_trajectory(
        initial_positions[0],
        initial_momenta[0],
        start_time,
        end_time,
        time_step,
        laser_parameters,
        pulse_parameters,
    )

    electron_positions = electron_positions.block_until_ready()
    electron_momenta = electron_momenta.block_until_ready()

    print("Integrating trajectory for all particles and computing scattered field")

    num_devices = len(jax.devices())
    # print(f"Mesh grid size: {num_devices}")

    mesh = jax.make_mesh((num_devices,), ("i",))
    with jax.set_mesh(mesh):
        batch_size = 1024
        num_batches = len(initial_positions) // batch_size

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
        np.arange(start_time, end_time + time_step / 2, time_step),
        np.linspace(
            central_frequency - frequency_width / 2,
            central_frequency + frequency_width / 2,
            num_frequencies,
        ),
        electron_positions,
        electron_momenta,
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
    # BUG: disable NVLink-related stuff; it gives error on our system
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    # Disable memory pre-allocation, since it's a shared system
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    print("Enabling float64 support in JAX")
    jax.config.update("jax_enable_x64", True)

    # print("Enabling CPU parallelism in JAX")
    # num_cpus = multiprocessing.cpu_count()
    # jax.config.update("jax_num_cpu_devices", min(128, num_cpus))

    # print("Available JAX devices:", jax.local_devices())

    jax.config.update("jax_compilation_cache_dir", "jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    # Monochromatic laser
    laser_frequency = 0.057
    # ~800 nm, red light
    laser_wavelength = (2 * pi * c) / laser_frequency

    a_0 = 1
    m_e = 1
    q = -1

    amplitude = a_0 * m_e * c * laser_frequency / abs(q)
    polarization = Polarizations.RIGHT_CIRCULAR.value
    waist_radius = 75 * laser_wavelength

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
    num_electrons = 16384

    print(f"Working with {num_electrons} electrons")

    disk_radius = (1.75 + radial_index) * waist_radius

    initial_positions = generate_initial_positions_on_disk(
        generator, disk_radius, num_electrons
    )
    # Add a 0 on the first index to obtain position 4-vectors
    initial_positions = np.concatenate(
        (np.zeros((num_electrons, 1), dtype=np.float64), initial_positions), axis=-1
    )

    # TODO: give the electrons initial velocities and check what is
    # the Doppler-shifted frequency of the scattered radiation (should be ~ 4 \gamma^2)

    initial_momenta = np.zeros((num_electrons, 4), dtype=np.float64)
    # u0 = c \gamma
    # We set \gamma to be 1 initially
    initial_momenta[:, 0] = c

    initial_positions = jnp.asarray(initial_positions)
    initial_momenta = jnp.asarray(initial_momenta)

    ### Pulse parameters ###
    tau_0 = 10 / laser_frequency
    phi_0 = 3 * tau_0

    pulse_parameters = GaussianPulseParameters(phi_0, tau_0)

    integration_start_time = 0.0
    integration_end_time = 6 * tau_0
    integration_duration = integration_end_time - integration_start_time

    time_step = integration_duration / 500

    print(
        f"Integrating from t = {integration_start_time} to t = {integration_end_time}, with a time step of dt = {time_step}"
    )

    central_frequency = laser_parameters.frequency
    frequency_width = 0.25 * central_frequency
    num_frequencies = 128

    detector_parameters = DetectorParameters(
        width=40 * 75 * laser_wavelength,
        height=40 * 75 * laser_wavelength,
        z_distance=-2 * 100_000 * laser_wavelength,
        grid_size_x=128,
        grid_size_y=128,
    )

    detector_positions = initialize_detector_positions(detector_parameters)

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
        detector_parameters,
        detector_positions,
    )

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Took {duration} seconds")

    print("Plotting results")

    plots_directory = Path("plots")
    plots_directory.mkdir(parents=True, exist_ok=True)

    ### Sample trajectory
    plot_sample_electron_trajectory(
        plots_directory,
        result.timestamps,
        np.asarray(result.electron_positions),
        np.asarray(result.electron_momenta),
    )

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
    detector_electric_field = result.detector_electric_field.reshape(
        detector_parameters.grid_size_y, detector_parameters.grid_size_x, -1
    )

    for index, coordinate in enumerate(("x", "y", "z")):
        fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

        fig.suptitle("Scattered electric field on detector")

        ax.set_title(f"$E_{coordinate}$")

        image = ax.imshow(np.real(detector_electric_field[:, :, index]), cmap="bwr")
        # plt.imshow(np.linalg.vector_norm(electric_field, axis=-1))
        # plt.imshow(np.angle(electric_field[:, :, 0]))
        # plt.legend()

        ax.set_xlabel("Detector $x$")
        ax.set_ylabel("Detector $y$")
        ax.grid()

        fig.colorbar(image)

        fig.tight_layout()
        fig.savefig(plots_directory / f"electric_field_{coordinate}.pdf")


def plot_sample_electron_trajectory(
    plots_directory: Path,
    timestamps: np.ndarray,
    positions: np.ndarray,
    momenta: np.ndarray,
) -> None:
    fig, ax = plt.subplots()

    # ax.plot(result.timestamps, positions[:, 0] - positions[:, 0].mean(), label="ct")
    ax.plot(timestamps, positions[:, 1] - positions[:, 1].mean(), label="x")
    ax.plot(timestamps, positions[:, 2] - positions[:, 2].mean(), label="y")
    ax.plot(timestamps, positions[:, 3] - positions[:, 3].mean(), label="z")

    ax.legend()
    ax.grid()

    fig.tight_layout()
    fig.savefig(plots_directory / "trajectory_positions.pdf")

    fig, ax = plt.subplots()

    gamma_ax = ax.twinx()
    gamma_ax.plot(timestamps, momenta[:, 0] / c, label="$\\gamma$", color="cyan")

    ax.plot(timestamps, momenta[:, 1], label="$u_1$")
    ax.plot(timestamps, momenta[:, 2], label="$u_2$")
    ax.plot(timestamps, momenta[:, 3], label="$u_3$")

    ax.legend()
    ax.grid()

    fig.tight_layout()
    fig.savefig(plots_directory / "trajectory_momenta.pdf")


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
