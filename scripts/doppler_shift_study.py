import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import typer
from rich import print

from electrodynamics.beams import LaguerreGaussBeamParameters
from electrodynamics.constants import ELECTRON_CHARGE, ELECTRON_MASS, SPEED_OF_LIGHT
from electrodynamics.fields import (
    compute_doppler_shift_factor,
    compute_scattered_electric_and_magnetic_fields,
)
from electrodynamics.initial_conditions import (
    generate_initial_particle_momenta_moving_towards_laser,
    generate_initial_positions_uniformly_on_disk,
)
from electrodynamics.integrate import integration_step_rk4
from electrodynamics.jax import initialize_jax
from electrodynamics.plotting import plot_scattered_radiation_spectrum
from electrodynamics.polarization import Polarizations
from electrodynamics.pulse import PulseWithFlatPeakParameters
from electrodynamics.serialization import ConfigurationEncoder

type PulseParameters = PulseWithFlatPeakParameters


@jdc.pytree_dataclass
class IntegrationResult:
    frequencies: np.ndarray

    scattered_electric_field_spectrum: jax.Array
    scattered_magnetic_field_spectrum: jax.Array


type IterationState = tuple[float, jax.Array, jax.Array, jax.Array, jax.Array]


@jax.shard_map(
    in_specs=(jax.P("i"),) * 2 + (None,) * 9,
    out_specs=(jax.P("i"),) * 2,
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
) -> tuple[jax.Array, jax.Array]:
    """Compute the trajectory and the scattered radiation from particles
    under the action of the laser pulse.
    """
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

    compute_scattered_fields_for_all_frequencies = jax.vmap(
        compute_scattered_electric_and_magnetic_fields,
        in_axes=(0, None, None, None, None),
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
        ) = u

        new_positions, new_momenta = integration_step_rk4(
            previous_positions,
            previous_momenta,
            time_step,
            laser_parameters,
            pulse_parameters,
        )

        electric_field, magnetic_field = compute_scattered_fields_for_all_frequencies(
            frequencies,
            new_positions,
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

        u_next = (
            proper_time + time_step,
            new_positions,
            new_momenta,
            new_scattered_electric_field,
            new_scattered_magnetic_field,
        )
        return u_next, None

    (
        (
            _final_time,
            _final_positions,
            _final_momenta,
            scattered_electric_field,
            scattered_magnetic_field,
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
        ),
        None,
        length=num_time_steps,
    )

    return (
        scattered_electric_field,
        scattered_magnetic_field,
    )


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
) -> IntegrationResult:
    print(
        "Simulating trajectories for all particles and computing frequency spectrum of scattered field"
    )

    num_devices = min(len(initial_positions), len(jax.devices()))

    mesh = jax.make_mesh((num_devices,), ("i",))
    with jax.set_mesh(mesh):
        batch_size = 1024
        num_batches = max(1, len(initial_positions) // batch_size)

        batches = zip(
            jnp.split(initial_positions, num_batches, axis=0),
            jnp.split(initial_momenta, num_batches, axis=0),
        )
        scattered_electric_field_spectrum = jnp.zeros((num_frequencies, 3))
        scattered_magnetic_field_spectrum = jnp.zeros((num_frequencies, 3))

        for index, batch in enumerate(batches):
            print(f"Batch #{index}")
            batch_positions = jax.device_put(batch[0], jax.P("i", None))
            batch_momenta = jax.device_put(batch[1], jax.P("i", None))

            (
                batch_scattered_electric_field_spectrum,
                batch_scattered_magnetic_field_spectrum,
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
            )

            scattered_electric_field_spectrum += jnp.sum(
                batch_scattered_electric_field_spectrum, axis=0
            ).block_until_ready()
            scattered_magnetic_field_spectrum += jnp.sum(
                batch_scattered_magnetic_field_spectrum, axis=0
            ).block_until_ready()

    return IntegrationResult(
        np.linspace(
            central_frequency - frequency_width / 2,
            central_frequency + frequency_width / 2,
            num_frequencies,
        ),
        scattered_electric_field_spectrum,
        scattered_magnetic_field_spectrum,
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
    num_electrons: Annotated[
        int, typer.Option(help="Number of electrons to simulate")
    ] = 4096,
    frequency_window_percentage: Annotated[
        float,
        typer.Option(
            help="Percentage of the central Doppler-shifted frequency around which to measure the spectrum."
        ),
    ] = 0.2,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Overwrite last run instead of creating a new directory, indexed with the timestamp of the current run."
        ),
    ] = False,
) -> None:
    """Simulates a Laguerre-Gauss laser pulse hitting a bunch of electrons moving at relativistic speeds.
    The scattered radiation is Doppler-shifted to much higher frequencies.

    We measure the frequency spectrum of the scattered radiation at a fixed point on the detector.
    """
    run_datetime = datetime.now(UTC).astimezone()
    print(f"Started run at {run_datetime}")

    initialize_jax()

    from math import pi

    c = SPEED_OF_LIGHT
    m_e = ELECTRON_MASS
    q = ELECTRON_CHARGE

    # Monochromatic laser
    laser_frequency = 0.057
    # ~800 nm, red light
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
    )

    seed = 42
    generator = np.random.default_rng(seed)
    disk_radius = 100 * laser_wavelength
    initial_positions = generate_initial_positions_uniformly_on_disk(
        generator, disk_radius, num_electrons
    )

    # Add a 0 on the first index to obtain position 4-vectors
    initial_positions = np.concatenate(
        (np.zeros((num_electrons, 1), dtype=np.float64), initial_positions), axis=-1
    )

    initial_momenta = generate_initial_particle_momenta_moving_towards_laser(
        num_particles=num_electrons, gamma=gamma, particle_mass=m_e
    )

    initial_positions = jnp.asarray(initial_positions)
    initial_momenta = jnp.asarray(initial_momenta)

    ### Pulse parameters ###
    tau_0 = (10 * np.pi) / laser_frequency
    phi_0 = 5 * tau_0
    peak_duration_periods = 10

    pulse_parameters = PulseWithFlatPeakParameters(phi_0, tau_0, peak_duration_periods)

    integration_start_time = 0.0
    stretching_factor = gamma + np.sqrt(gamma**2 - 1)
    integration_end_time = (
        10 * tau_0 + peak_duration_periods * tau_0
    ) / stretching_factor
    time_step = ((2 * pi) / laser_frequency) / 100 / stretching_factor

    print(
        f"Integrating from t = {integration_start_time} to t = {integration_end_time}, with a time step of dt = {time_step}"
    )

    doppler_shift_factor = compute_doppler_shift_factor(gamma, m_e, q, a_0)

    central_frequency = laser_parameters.frequency * doppler_shift_factor
    print(f"Looking around Fourier frequency omega = {central_frequency}")

    frequency_width = frequency_window_percentage * central_frequency
    num_frequencies = 256

    z_distance = -25_000 * laser_wavelength
    spectrum_measurement_position = jnp.array(
        (
            15 * laser_wavelength,
            15 * laser_wavelength,
            z_distance,
        )
    )
    print(
        f"Measuring scattered radiation frequency spectrum at {spectrum_measurement_position}"
    )

    runs_directory = Path("runs")

    if overwrite:
        output_directory = runs_directory / "doppler_shift_latest"
    else:
        timestamp = run_datetime.strftime("%Y-%d-%m_%H-%M-%S")
        output_directory = runs_directory / f"doppler_shift_{timestamp}"

    output_directory.mkdir(parents=True, exist_ok=True)

    print("Saving run configuration")
    configuration = {
        "num_electrons": num_electrons,
        "a_0": a_0,
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

    with open(output_directory / "configuration.json", "w") as file:
        json.dump(configuration, file, indent=2, cls=ConfigurationEncoder)

    plots_directory = output_directory

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
    )

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Took {duration} seconds")

    print("Plotting experimental setup")

    ### Scattered electric field spectrum
    fig = plt.figure(dpi=200)

    plot_scattered_radiation_spectrum(
        fig,
        central_frequency,
        result.frequencies,
        np.asarray(result.scattered_electric_field_spectrum),
    )

    fig.savefig(plots_directory / "electric_field_spectrum.pdf")


if __name__ == "__main__":
    typer.run(main)
