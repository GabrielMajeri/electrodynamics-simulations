from datetime import UTC, datetime
from math import pi
from pathlib import Path
from time import perf_counter
from typing import Annotated, cast

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import typer
from rich import print

from electrodynamics.beams import LaguerreGaussBeamParameters
from electrodynamics.constants import (
    DEFAULT_LASER_FREQUENCY,
)
from electrodynamics.constants import (
    ELECTRON_CHARGE as q,
)
from electrodynamics.constants import (
    ELECTRON_MASS as m_e,
)
from electrodynamics.constants import (
    SPEED_OF_LIGHT as c,
)
from electrodynamics.fields import compute_doppler_shift_factor
from electrodynamics.initial_conditions import (
    generate_initial_particle_momenta_moving_towards_laser,
    generate_initial_positions_uniformly_on_disk,
    to_position_four_vectors,
)
from electrodynamics.integrate import integration_step_rk4
from electrodynamics.jax import initialize_jax
from electrodynamics.plotting import save_figure
from electrodynamics.polarization import Polarizations
from electrodynamics.pulse import (
    PulseWithFlatPeakParameters,
    gaussian_envelope_with_flat_peak,
)
from electrodynamics.serialization import prepare_output_directory, save_json


@jdc.pytree_dataclass
class ParticleTrajectory:
    timestamps: jax.Array
    positions: jax.Array
    momenta: jax.Array

    oscillatory_kernel_exponents: jax.Array
    electric_field_multipliers: jax.Array
    magnetic_field_multipliers: jax.Array


@jdc.jit
def compute_particle_trajectory(
    initial_position: jax.Array,
    initial_momentum: jax.Array,
    start_time: jdc.Static[float],
    end_time: jdc.Static[float],
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseWithFlatPeakParameters],
    central_frequency: jdc.Static[float],
    spectrum_measurement_position: jax.Array,
) -> ParticleTrajectory:
    "Compute the trajectory of a single particle under the action of the laser pulse."

    num_time_steps = int((end_time - start_time) / time_step) + 1

    def scan_fn(
        u: tuple[float, jax.Array, jax.Array, float, jax.Array, jax.Array], _: None
    ) -> tuple[
        tuple[float, jax.Array, jax.Array, float, jax.Array, jax.Array],
        tuple[float, jax.Array, jax.Array, float, jax.Array, jax.Array],
    ]:
        (
            proper_time,
            previous_position,
            previous_momentum,
            _previous_oscillatory_kernel_exponent,
            _previous_electric_field_component,
            _previous_magnetic_field_component,
        ) = u

        new_position, new_momentum = integration_step_rk4(
            previous_position,
            previous_momentum,
            time_step,
            laser_parameters,
            pulse_parameters,
        )

        laboratory_time = cast(float, new_position[0, 0] / c)
        particle_position = new_position[0, 1:4]
        particle_velocity = new_momentum[0, 1:4]

        particle_displacement = particle_position - initial_position[1:4]
        detector_displacement = spectrum_measurement_position - initial_position[1:4]

        displacement = detector_displacement - particle_displacement
        displacement_norm = cast(float, jnp.linalg.vector_norm(displacement, axis=-1))
        view_direction = displacement / displacement_norm

        oscillatory_kernel_exponent = central_frequency * (
            laboratory_time + displacement_norm / c
        )

        beta = particle_velocity / c
        gamma_factor = cast(float, new_momentum[0, 0] / c)

        electric_field_common_term = jnp.cross(
            view_direction, jnp.cross(view_direction, beta)
        )

        electric_field_first_term = -((1j * central_frequency) / c) * (
            electric_field_common_term / displacement_norm
        )

        displacement_norm_squared = displacement_norm * displacement_norm

        electric_field_second_term = (
            electric_field_common_term
            + view_direction
            * jnp.expand_dims(
                gamma_factor + jnp.linalg.vecdot(view_direction, beta), axis=-1
            )
        ) / displacement_norm_squared

        n_cross_beta = jnp.cross(view_direction, beta)

        magnetic_field_first_term = ((1j * central_frequency) / c) * (
            n_cross_beta / displacement_norm
        )
        magnetic_field_second_term = n_cross_beta / displacement_norm_squared

        electric_field_multiplier = (
            electric_field_first_term + electric_field_second_term
        )
        magnetic_field_multiplier = (
            magnetic_field_first_term + magnetic_field_second_term
        )

        u_next = (
            proper_time + time_step,
            jnp.squeeze(new_position),
            jnp.squeeze(new_momentum),
            oscillatory_kernel_exponent,
            electric_field_multiplier,
            magnetic_field_multiplier,
        )
        return u_next, u_next

    _, trajectory = jax.lax.scan(
        scan_fn,
        (
            start_time,
            initial_position,
            initial_momentum,
            0,
            jnp.zeros(shape=3, dtype=jnp.complex128),
            jnp.zeros(shape=3, dtype=jnp.complex128),
        ),
        None,
        length=num_time_steps,
    )

    (
        timestamps,
        positions,
        momenta,
        oscillatory_kernel_exponents,
        electric_field_multipliers,
        magnetic_field_multipliers,
    ) = trajectory
    return ParticleTrajectory(
        jnp.asarray(timestamps),
        positions,
        momenta,
        jnp.asarray(oscillatory_kernel_exponents),
        electric_field_multipliers,
        magnetic_field_multipliers,
    )


def main(
    radial_index: Annotated[
        int, typer.Option(help="Integer counting radial nodes of the LG beam profile")
    ] = 2,
    azimuthal_index: Annotated[
        int,
        typer.Option(
            help="Integer controlling the on-axis phase singularity (topological charge)."
        ),
    ] = -2,
    a_0: Annotated[float, typer.Option(help="Normalized laser intensity")] = 1e-1,
    gamma: Annotated[float, typer.Option(help="Electrons' relativistic factor")] = 1,
    seed: Annotated[int, typer.Option(help="Seed for random number generator")] = 42,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Overwrite last run instead of creating a new directory, indexed with the timestamp of the current run."
        ),
    ] = False,
    exact_paraxial: Annotated[
        bool,
        typer.Option(
            help="Use exact paraxial derivatives for longitudinal EM fields (`E_z`, `B_z`) instead of envelope-only approximation."
        ),
    ] = True,
) -> None:
    "Computes the trajectory and the scattered electric field for a single electron."

    if radial_index < 0:
        print("`radial_index` must be a non-negative integer")
        raise typer.Exit(1)

    run_datetime = datetime.now(UTC).astimezone()
    print(f"Started run at {run_datetime}")

    print("Configuring and initializing JAX")
    initialize_jax()

    laser_frequency = DEFAULT_LASER_FREQUENCY
    laser_wavelength = (2 * pi * c) / laser_frequency

    amplitude = a_0 * m_e * c * laser_frequency / abs(q)
    polarization = Polarizations.RIGHT_CIRCULAR.value
    waist_radius = 75 * laser_wavelength

    laser_parameters = LaguerreGaussBeamParameters(
        laser_frequency,
        laser_wavelength,
        amplitude,
        polarization,
        waist_radius,
        radial_index,
        azimuthal_index,
        exact_paraxial,
    )
    print("Laser parameters:", laser_parameters)

    doppler_shift_factor = compute_doppler_shift_factor(gamma, m_e, q, a_0)
    print(f"Relativistic factor (gamma): {gamma}")
    print(f"Corresponding Doppler shift: {doppler_shift_factor}")

    central_frequency = laser_parameters.frequency * doppler_shift_factor
    print(f"Looking around Fourier frequency omega = {central_frequency}")

    generator = np.random.default_rng(seed)
    disk_radius = 10 * laser_wavelength

    initial_position_3d = generate_initial_positions_uniformly_on_disk(
        generator, disk_radius, 1
    )
    initial_position = to_position_four_vectors(initial_position_3d)
    initial_momentum = generate_initial_particle_momenta_moving_towards_laser(
        1, gamma, m_e
    )

    initial_position = jnp.asarray(initial_position[0])
    initial_momentum = jnp.asarray(initial_momentum[0])

    tau_0 = (10 * jnp.pi) / laser_frequency
    phi_0 = 5 * tau_0
    peak_duration_periods = 10
    pulse_parameters = PulseWithFlatPeakParameters(phi_0, tau_0, peak_duration_periods)
    print("Pulse parameters:", pulse_parameters)

    integration_start_time = 0.0
    stretching_factor = gamma + np.sqrt(gamma**2 - 1)
    integration_end_time = (
        10 * tau_0 + peak_duration_periods * tau_0
    ) / stretching_factor
    time_step = ((2 * pi) / laser_frequency) / 100 / stretching_factor

    print(
        f"Integrating trajectory from t_initial = {integration_start_time}, up to t_final = {integration_end_time}, with a time step of dt = {time_step}"
    )

    detector_z_distance = -25_000 * laser_wavelength
    spectrum_measurement_position = jnp.array(
        (
            25 * laser_parameters.wavelength,
            -25 * laser_parameters.wavelength,
            detector_z_distance,
        )
    )

    print("Starting numerical integration...")
    start_time = perf_counter()

    trajectory = compute_particle_trajectory(
        initial_position,
        initial_momentum,
        integration_start_time,
        integration_end_time,
        time_step,
        laser_parameters,
        pulse_parameters,
        central_frequency,
        spectrum_measurement_position,
    )

    end_time = perf_counter()
    print(f"Took {end_time - start_time} seconds")

    print("Saving results to disk")
    output_directory = prepare_output_directory(
        "single_electron_trajectory", run_datetime, overwrite=overwrite
    )

    configuration = {
        "laser_parameters": laser_parameters,
        "pulse_parameters": pulse_parameters,
        "electrons_initial_gamma": gamma,
        "time_start": integration_start_time,
        "time_end": integration_end_time,
        "time_end_in_multiples_of_tau_0": f"{integration_end_time / tau_0} * tau_0",
        "time_delta": time_step,
        "spectrum_measurement_position": spectrum_measurement_position,
        "spectrum_measurement_position_in_multiples_of_lambda": spectrum_measurement_position
        / laser_wavelength,
    }
    save_json(output_directory / "configuration.json", configuration)

    fig, ax = plt.subplots(dpi=200)
    fig.suptitle("Laser pulse envelope")
    phi = jnp.linspace(0, 10 * tau_0 + peak_duration_periods * tau_0, 1000)
    envelope = jnp.abs(
        jnp.exp(1j * laser_parameters.frequency * phi)
        * gaussian_envelope_with_flat_peak(
            phi,
            pulse_parameters.phi_0,
            pulse_parameters.tau_0,
            pulse_parameters.peak_duration_periods,
        )
    )
    ax.plot(phi / ((2 * jnp.pi) / laser_frequency), envelope)
    ax.set_xlabel("$\\phi$ (periods)")
    ax.set_ylabel("Envelope")
    ax.grid()
    fig.tight_layout()
    save_figure(fig, output_directory / "laser_pulse_envelope", formats=("pdf",))

    plot_electron_trajectory(
        output_directory,
        np.asarray(trajectory.timestamps),
        np.asarray(trajectory.positions),
        np.asarray(trajectory.momenta),
    )

    fig, ax = plt.subplots(dpi=200)
    fig.suptitle("Exponent of the oscillatory kernel")
    ax.set_title("$\\omega (t + R(x_0, t)/c)$")
    ax.plot(trajectory.timestamps, trajectory.oscillatory_kernel_exponents)
    oscillatory_kernel_derivative = jnp.gradient(
        trajectory.oscillatory_kernel_exponents, trajectory.timestamps
    )
    twin_ax = ax.twinx()
    twin_ax.plot(trajectory.timestamps, oscillatory_kernel_derivative)
    ax.grid()
    fig.tight_layout()
    save_figure(fig, output_directory / "oscillatory_kernel_exponents", formats=("pdf",))

    fig, ax = plt.subplots(dpi=200)
    fig.suptitle("Oscillatory kernel")
    ax.set_title("$\\exp(i \\omega (t + R(x_0, t)/c))$")
    ax.plot(
        trajectory.timestamps,
        jnp.real(jnp.exp(1j * trajectory.oscillatory_kernel_exponents)),
    )
    ax.grid()
    fig.tight_layout()
    save_figure(fig, output_directory / "oscillatory_kernel", formats=("pdf",))

    fig, ax = plt.subplots(dpi=200)
    fig.suptitle("Multiplier in the electric field integral")
    E_x, E_y, E_z = np.real(trajectory.electric_field_multipliers).T
    ax.plot(trajectory.timestamps, E_x - np.mean(E_x), label="$E_x$")
    ax.plot(trajectory.timestamps, E_y - np.mean(E_y), label="$E_y$")
    ax.plot(trajectory.timestamps, E_z - np.mean(E_z), label="$E_z$")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    save_figure(fig, output_directory / "electric_field_multipliers", formats=("pdf",))

    fig, ax = plt.subplots(dpi=200)
    fig.suptitle("Multiplier in the magnetic field integral")
    B_x, B_y, B_z = np.real(trajectory.magnetic_field_multipliers).T
    ax.plot(trajectory.timestamps, B_x - np.mean(B_x), label="$B_x$")
    ax.plot(trajectory.timestamps, B_y - np.mean(B_y), label="$B_y$")
    ax.plot(trajectory.timestamps, B_z - np.mean(B_z), label="$B_z$")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    save_figure(fig, output_directory / "magnetic_field_multipliers", formats=("pdf",))

    print(f"Saved run output to '{output_directory}'")


def plot_electron_trajectory(
    plots_directory: Path,
    timestamps: np.ndarray,
    positions: np.ndarray,
    momenta: np.ndarray,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(timestamps, positions[:, 1] - positions[:, 1].mean(), label="x")
    ax.plot(timestamps, positions[:, 2] - positions[:, 2].mean(), label="y")
    ax.plot(timestamps, positions[:, 3] - positions[:, 3].mean(), label="z")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    save_figure(fig, plots_directory / "trajectory_positions", formats=("pdf",))

    fig, ax = plt.subplots()
    gamma_ax = ax.twinx()
    gamma_ax.plot(timestamps, momenta[:, 0] / c, label="$\\gamma$", color="cyan")
    ax.plot(timestamps, momenta[:, 1], label="$u_1$")
    ax.plot(timestamps, momenta[:, 2], label="$u_2$")
    ax.plot(timestamps, momenta[:, 3], label="$u_3$")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    save_figure(fig, plots_directory / "trajectory_momenta", formats=("pdf",))


if __name__ == "__main__":
    typer.run(main)
