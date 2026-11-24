from datetime import UTC, datetime
from math import pi
from time import perf_counter
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax.sharding import PartitionSpec as P
from rich import print

from electrodynamics.angular_momentum import compute_particle_angular_momentum_z
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
from electrodynamics.detector import (
    AnyDetectorParameters,
    build_detector_grid,
    create_detector_parameters,
)
from electrodynamics.energy import (
    analyze_energy_transfer,
    build_energy_summary_dict,
    build_frequency_quadrature,
    compute_detector_spectral_energy,
    compute_detector_total_energy,
    compute_electron_kinetic_energy_balance,
    compute_factorized_detector_energy,
    compute_incident_beam_energy,
    compute_larmor_radiated_energy_step,
    format_energy_transfer_report,
)
from electrodynamics.fields import (
    compute_coherent_detector_fields_multifreq,
    compute_doppler_shift_factor,
    compute_scattered_electric_and_magnetic_fields,
)
from electrodynamics.initial_conditions import (
    generate_initial_particle_momenta_at_rest,
    generate_initial_particle_momenta_moving_towards_laser,
    generate_initial_positions_in_nanofiber,
    generate_initial_positions_on_thin_sheet,
    generate_initial_positions_uniformly_on_disk,
    generate_initial_positions_uniformly_within_ball,
    to_position_four_vectors,
)
from electrodynamics.integrate import integration_step_rk4
from electrodynamics.jax import initialize_jax, iterate_sharded_particle_batches
from electrodynamics.plotting import (
    plot_angular_momentum_distribution,
    plot_experimental_setup,
    plot_final_momentum_distribution,
    plot_scattered_radiation_spectrum,
    save_detector_field_component_plots,
    save_figure,
)
from electrodynamics.polarization import Polarizations
from electrodynamics.pulse import PulseWithFlatPeakParameters
from electrodynamics.serialization import prepare_output_directory, save_json

type PulseParameters = PulseWithFlatPeakParameters

type IterationState = tuple[
    float,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


@jax.shard_map(
    in_specs=(P("i"),) * 2 + (None,) * 12,
    out_specs=(P("i"),) * 7 + (P(), P()),
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
    detector_parameters: jdc.Static[AnyDetectorParameters],
    detector_positions: jdc.Static[jax.Array],
    energy_frequencies: jdc.Static[jax.Array],
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Compute the trajectory and the scattered radiation from particles
    under the action of the laser pulse.
    """
    frequencies = jnp.linspace(
        central_frequency - frequency_width / 2,
        central_frequency + frequency_width / 2,
        num_frequencies,
    )

    batch_size = len(initial_positions)

    scattered_electric_field = jax.lax.pcast(
        jnp.zeros(shape=(batch_size, num_frequencies, 3), dtype=jnp.complex128),
        "i",
        to="varying",
    )
    scattered_magnetic_field = jax.lax.pcast(
        jnp.zeros(shape=(batch_size, num_frequencies, 3), dtype=jnp.complex128),
        "i",
        to="varying",
    )

    num_detector_points = len(detector_positions)
    detector_electric_field = jax.lax.pcast(
        jnp.zeros(shape=(batch_size, num_detector_points, 3), dtype=jnp.complex128),
        "i",
        to="varying",
    )
    detector_magnetic_field = jax.lax.pcast(
        jnp.zeros(shape=(batch_size, num_detector_points, 3), dtype=jnp.complex128),
        "i",
        to="varying",
    )
    larmor_energy = jax.lax.pcast(
        jnp.zeros(shape=(batch_size,), dtype=jnp.float64),
        "i",
        to="varying",
    )
    num_energy_frequencies = len(energy_frequencies)
    detector_electric_field_multifreq = jax.lax.pcast(
        jnp.zeros(
            shape=(num_detector_points, num_energy_frequencies, 3), dtype=jnp.complex128
        ),
        "i",
        to="varying",
    )
    detector_magnetic_field_multifreq = jax.lax.pcast(
        jnp.zeros(
            shape=(num_detector_points, num_energy_frequencies, 3), dtype=jnp.complex128
        ),
        "i",
        to="varying",
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
            previous_larmor_energy,
            previous_detector_electric_field_mf,
            previous_detector_magnetic_field_mf,
        ) = u

        new_positions, new_momenta = integration_step_rk4(
            previous_positions,
            previous_momenta,
            time_step,
            laser_parameters,
            pulse_parameters,
        )

        electric_field, magnetic_field = jax.vmap(
            lambda frequency: compute_scattered_electric_and_magnetic_fields(
                frequency,
                new_positions,
                new_momenta,
                initial_positions,
                spectrum_measurement_position,
                initial_momentum=initial_momenta,
            ),
            in_axes=0,
            out_axes=1,
        )(frequencies)
        new_scattered_electric_field = (
            previous_scattered_electric_field + time_step * electric_field
        )
        new_scattered_magnetic_field = (
            previous_scattered_magnetic_field + time_step * magnetic_field
        )

        electric_field, magnetic_field = jax.vmap(
            lambda det_pos: compute_scattered_electric_and_magnetic_fields(
                central_frequency,
                new_positions,
                new_momenta,
                initial_positions,
                det_pos,
                initial_momentum=initial_momenta,
            ),
            in_axes=0,
            out_axes=1,
        )(detector_positions)

        new_detector_electric_field = (
            previous_detector_electric_field + time_step * electric_field
        )
        new_detector_magnetic_field = (
            previous_detector_magnetic_field + time_step * magnetic_field
        )
        new_larmor_energy = (
            previous_larmor_energy
            + compute_larmor_radiated_energy_step(
                new_positions,
                new_momenta,
                time_step,
                laser_parameters,
                pulse_parameters,
                is_lab_time=False,
            )
        )

        if num_energy_frequencies > 0:
            dE_mf, dB_mf = compute_coherent_detector_fields_multifreq(
                new_positions,
                new_momenta,
                initial_positions,
                detector_positions,
                energy_frequencies,
                is_lab_time=False,
                initial_momentum=initial_momenta,
            )
            new_detector_electric_field_mf = (
                previous_detector_electric_field_mf + time_step * dE_mf
            )
            new_detector_magnetic_field_mf = (
                previous_detector_magnetic_field_mf + time_step * dB_mf
            )
        else:
            new_detector_electric_field_mf = previous_detector_electric_field_mf
            new_detector_magnetic_field_mf = previous_detector_magnetic_field_mf

        u_next = (
            proper_time + time_step,
            new_positions,
            new_momenta,
            new_scattered_electric_field,
            new_scattered_magnetic_field,
            new_detector_electric_field,
            new_detector_magnetic_field,
            new_larmor_energy,
            new_detector_electric_field_mf,
            new_detector_magnetic_field_mf,
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
            larmor_energy,
            detector_electric_field_multifreq,
            detector_magnetic_field_multifreq,
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
            larmor_energy,
            detector_electric_field_multifreq,
            detector_magnetic_field_multifreq,
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
        larmor_energy,
        jax.lax.psum(detector_electric_field_multifreq, "i"),
        jax.lax.psum(detector_magnetic_field_multifreq, "i"),
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

    larmor_energy_per_particle: jax.Array
    detector_electric_field_multifreq: jax.Array
    detector_magnetic_field_multifreq: jax.Array


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
    detector_parameters: jdc.Static[AnyDetectorParameters],
    detector_positions: jdc.Static[jax.Array],
    energy_frequencies: jax.Array,
    batch_size: jdc.Static[int],
) -> IntegrationResult:
    print("Integrating trajectory for all particles and computing scattered field")

    final_positions = []
    final_momenta = []
    larmor_energies = []
    scattered_electric_field_spectrum = jnp.zeros(
        (num_frequencies, 3), dtype=jnp.complex128
    )
    scattered_magnetic_field_spectrum = jnp.zeros(
        (num_frequencies, 3), dtype=jnp.complex128
    )
    detector_electric_field = jnp.zeros(
        (len(detector_positions), 3), dtype=jnp.complex128
    )
    detector_magnetic_field = jnp.zeros(
        (len(detector_positions), 3), dtype=jnp.complex128
    )
    detector_electric_field_multifreq = jnp.zeros(
        (len(detector_positions), len(energy_frequencies), 3), dtype=jnp.complex128
    )
    detector_magnetic_field_multifreq = jnp.zeros(
        (len(detector_positions), len(energy_frequencies), 3), dtype=jnp.complex128
    )

    for (
        index,
        _num_batches,
        batch_positions,
        batch_momenta,
    ) in iterate_sharded_particle_batches(
        initial_positions, initial_momenta, batch_size
    ):
        print(f"Batch #{index}")
        (
            batch_final_positions,
            batch_final_momenta,
            batch_scattered_electric_field_spectrum,
            batch_scattered_magnetic_field_spectrum,
            batch_detector_electric_field,
            batch_detector_magnetic_field,
            batch_larmor_energy,
            batch_detector_electric_field_mf,
            batch_detector_magnetic_field_mf,
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
            energy_frequencies,
        )

        final_positions.append(batch_final_positions.block_until_ready())
        final_momenta.append(batch_final_momenta.block_until_ready())
        larmor_energies.append(batch_larmor_energy.block_until_ready())
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
        detector_electric_field_multifreq += (
            batch_detector_electric_field_mf.block_until_ready()
        )
        detector_magnetic_field_multifreq += (
            batch_detector_magnetic_field_mf.block_until_ready()
        )

    return IntegrationResult(
        np.arange(start_time, end_time + 1e-10, time_step),
        np.linspace(
            central_frequency - frequency_width / 2,
            central_frequency + frequency_width / 2,
            num_frequencies,
        ),
        jnp.concatenate(final_positions),
        jnp.concatenate(final_momenta),
        scattered_electric_field_spectrum,
        scattered_magnetic_field_spectrum,
        detector_electric_field,
        detector_magnetic_field,
        jnp.concatenate(larmor_energies),
        detector_electric_field_multifreq,
        detector_magnetic_field_multifreq,
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
    exact_paraxial: Annotated[
        bool,
        typer.Option(
            help="Use exact paraxial derivatives for longitudinal EM fields (`E_z`, `B_z`) instead of envelope-only approximation."
        ),
    ] = True,
    gamma: Annotated[float, typer.Option(help="Electrons' relativistic factor")] = 1,
    num_electrons: Annotated[
        int, typer.Option(help="Number of electrons to simulate")
    ] = 1024,
    target_type: Annotated[
        str,
        typer.Option(
            help="Geometry for initial electron distribution: 'disk', 'thin-sheet', 'ball', or 'nanofiber'."
        ),
    ] = "disk",
    batch_size: Annotated[
        int, typer.Option(help="Number of electrons to simulate in each batch")
    ] = 512,
    frequency_window_percentage: Annotated[
        float,
        typer.Option(
            help="Percentage of the central Doppler-shifted frequency around which to measure the spectrum."
        ),
    ] = 0.2,
    detector_shape: Annotated[
        Literal["rect", "disk"],
        typer.Option(help="Detector geometry: 'rect' (Cartesian) or 'disk' (polar)"),
    ] = "rect",
    detector_grid_size: Annotated[
        int, typer.Option(help="Detector resolution per axis for 'rect' detector")
    ] = 64,
    detector_num_radial: Annotated[
        int, typer.Option(help="Number of radial rings for 'disk' detector")
    ] = 48,
    detector_num_azimuthal: Annotated[
        int, typer.Option(help="Number of azimuthal samples for 'disk' detector")
    ] = 64,
    detector_radial_sampling: Annotated[
        Literal["gauss-legendre", "uniform", "equal-area"],
        typer.Option(
            help="Radial quadrature scheme for 'disk' detector: 'gauss-legendre', 'uniform', or 'equal-area'"
        ),
    ] = "gauss-legendre",
    energy_num_frequencies: Annotated[
        int,
        typer.Option(
            help="Number of Gauss-Legendre frequency nodes for total detector energy (0 to use factorized estimate only)"
        ),
    ] = 16,
    energy_bandwidth_fraction: Annotated[
        float,
        typer.Option(
            help="Full fractional bandwidth Delta_omega / omega_1 for total detector energy quadrature"
        ),
    ] = 0.6,
    physical_electrons: Annotated[
        float | None,
        typer.Option(
            help="Optional physical electron count N_real for coherent N_e^2 bunch extrapolation"
        ),
    ] = None,
    physical_bunch_charge_pc: Annotated[
        float | None,
        typer.Option(
            help="Optional physical bunch charge in pC for coherent N_e^2 extrapolation"
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Overwrite last run instead of creating a new directory, indexed with the timestamp of the current run."
        ),
    ] = False,
) -> None:
    """Simulates a Laguerre-Gauss laser pulse hitting a bunch of electrons moving at relativistic speeds.
    The scattered radiation is Doppler-shifted to much higher frequencies.

    We compute the spectrum of the integrated radiation received on a rectangular detector positioned far-away from the particles.
    """
    run_datetime = datetime.now(UTC).astimezone()
    print(f"Started run at {run_datetime}")

    if energy_num_frequencies < 0:
        raise ValueError(
            f"energy_num_frequencies must be >= 0, got {energy_num_frequencies}"
        )
    if physical_electrons is not None and physical_bunch_charge_pc is not None:
        raise ValueError(
            "Specify at most one of --physical-electrons or --physical-bunch-charge-pc."
        )

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

    seed = 42
    generator = np.random.default_rng(seed)
    print(f"Working with {num_electrons} electrons (target geometry: {target_type})")

    disk_radius = 100 * laser_wavelength

    if target_type == "disk":
        initial_positions_3d = generate_initial_positions_uniformly_on_disk(
            generator, disk_radius, num_electrons
        )
    elif target_type == "thin-sheet":
        initial_positions_3d = generate_initial_positions_on_thin_sheet(
            disk_radius=disk_radius,
            num_particles=num_electrons,
            sampling_method="random",
            generator=generator,
        )
    elif target_type == "ball":
        initial_positions_3d = generate_initial_positions_uniformly_within_ball(
            generator, disk_radius, num_electrons
        )
    elif target_type == "nanofiber":
        fiber_radius = 0.5 * laser_wavelength
        fiber_length = 15.0 * laser_wavelength
        initial_positions_3d = generate_initial_positions_in_nanofiber(
            generator, fiber_radius, fiber_length, num_electrons
        )
    else:
        raise ValueError(
            f"Unknown target_type '{target_type}'. Expected 'disk', 'thin-sheet', 'ball', or 'nanofiber'."
        )

    initial_positions = to_position_four_vectors(initial_positions_3d)

    if target_type == "nanofiber" and np.isclose(gamma, 1.0):
        initial_momenta = generate_initial_particle_momenta_at_rest(num_electrons, m_e)
    else:
        initial_momenta = generate_initial_particle_momenta_moving_towards_laser(
            num_particles=num_electrons, gamma=gamma, particle_mass=m_e
        )

    initial_positions = jnp.asarray(initial_positions)
    initial_momenta = jnp.asarray(initial_momenta)

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
    detector_parameters = create_detector_parameters(
        detector_shape,
        span=10 * 75 * laser_wavelength,
        grid_size=detector_grid_size,
        num_radial=detector_num_radial,
        num_azimuthal=detector_num_azimuthal,
        radial_sampling=detector_radial_sampling,
    )
    detector_grid = build_detector_grid(
        detector_parameters, distance=abs(z_distance), direction=-1
    )
    detector_positions = jnp.asarray(detector_grid.lab_positions_flat)

    spectrum_measurement_position = jnp.array(
        (
            25 * laser_wavelength,
            -25 * laser_wavelength,
            z_distance,
        )
    )

    energy_quadrature = (
        build_frequency_quadrature(
            central_frequency,
            bandwidth_fraction=energy_bandwidth_fraction,
            num_frequencies=energy_num_frequencies,
        )
        if energy_num_frequencies > 0
        else None
    )
    energy_frequencies_jnp = (
        jnp.asarray(energy_quadrature.frequencies)
        if energy_quadrature is not None
        else jnp.zeros((0,), dtype=jnp.float64)
    )

    output_directory = prepare_output_directory(
        "oam_transfer", run_datetime, overwrite=overwrite
    )

    print("Saving run configuration")
    configuration = {
        "laser_parameters": laser_parameters,
        "pulse_parameters": pulse_parameters,
        "num_electrons": num_electrons,
        "target_type": target_type,
        "electrons_initial_gamma": gamma,
        "time_start": integration_start_time,
        "time_end": integration_end_time,
        "time_end_in_multiples_of_tau_0": f"{integration_end_time / tau_0} * tau_0",
        "time_delta": time_step,
        "detector_shape": detector_shape,
        "detector_parameters": detector_parameters,
        "energy_num_frequencies": energy_num_frequencies,
        "energy_bandwidth_fraction": energy_bandwidth_fraction,
        "physical_electrons": physical_electrons,
        "physical_bunch_charge_pc": physical_bunch_charge_pc,
        "spectrum_measurement_position": spectrum_measurement_position,
        "spectrum_measurement_position_in_multiples_of_lambda": spectrum_measurement_position
        / laser_wavelength,
    }
    save_json(output_directory / "configuration.json", configuration)

    print("Plotting experimental setup")
    fig = plt.figure(figsize=(8, 5), dpi=300)
    plot_experimental_setup(
        fig,
        np.asarray(initial_positions),
        np.asarray(initial_momenta),
        detector_parameters,
        np.asarray(detector_positions),
    )
    save_figure(fig, output_directory / "experimental_setup", formats=("pdf",))

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
        energy_frequencies_jnp,
        batch_size,
    )

    end_time = perf_counter()
    print(f"Took {end_time - start_time} seconds")

    print("Plotting results")

    fig = plt.figure(dpi=200)
    plot_final_momentum_distribution(
        fig,
        np.asarray(initial_positions[:, 1:4]),
        waist_radius,
        np.asarray(result.final_momenta[:, 3]),
    )
    save_figure(fig, output_directory / "final_momenta")

    fig = plt.figure(dpi=200)
    angular_momenta = compute_particle_angular_momentum_z(
        result.final_positions, result.final_momenta
    )
    plot_angular_momentum_distribution(
        fig,
        np.asarray(initial_positions[:, 1:4]),
        waist_radius,
        np.asarray(angular_momenta),
    )
    save_figure(fig, output_directory / "final_angular_momenta")

    fig = plt.figure(dpi=200)
    plot_scattered_radiation_spectrum(
        fig,
        central_frequency,
        result.frequencies,
        np.asarray(result.scattered_electric_field_spectrum),
    )
    save_figure(fig, output_directory / "electric_field_spectrum", formats=("pdf",))

    detector_electric_field = detector_parameters.reshape_to_grid(
        np.asarray(result.detector_electric_field)
    )
    detector_magnetic_field = detector_parameters.reshape_to_grid(
        np.asarray(result.detector_magnetic_field)
    )
    save_detector_field_component_plots(
        output_directory,
        detector_electric_field,
        detector_parameters,
        waist_radius,
        formats=("pdf",),
        local_coords_2d=detector_grid.local_coords_2d,
    )

    np.save(
        output_directory / "detector_area_weights.npy", detector_grid.area_weights_2d
    )

    spec_E_np = np.asarray(result.scattered_electric_field_spectrum)
    detector_spectral_energy = compute_detector_spectral_energy(
        detector_electric_field,
        detector_magnetic_field,
        detector_grid.area_weights_2d,
        normal_vector=detector_grid.normal_local,
        detector_positions=detector_grid.local_coords_2d,
    )
    factorized_energy = compute_factorized_detector_energy(
        detector_electric_field,
        detector_magnetic_field,
        detector_grid.area_weights_2d,
        result.frequencies,
        spec_E_np,
        reference_frequency=central_frequency,
        normal_vector=detector_grid.normal_local,
    )

    if energy_quadrature is not None:
        det_E_mf_grid = detector_parameters.reshape_to_grid(
            np.asarray(result.detector_electric_field_multifreq)
        )
        det_B_mf_grid = detector_parameters.reshape_to_grid(
            np.asarray(result.detector_magnetic_field_multifreq)
        )
        detector_total_energy = compute_detector_total_energy(
            det_E_mf_grid,
            det_B_mf_grid,
            detector_grid.area_weights_2d,
            energy_quadrature.weights,
            normal_vector=detector_grid.normal_lab,
            detector_positions=detector_grid.reshape_to_grid(
                detector_grid.lab_positions_flat
            ),
        )
        detector_energy_au = detector_total_energy.total_energy_au
        energy_method = "multifreq-quadrature"
        np.save(
            output_directory / "detector_fluence.npy",
            detector_total_energy.fluence_per_pixel,
        )
        np.save(
            output_directory / "detector_energy_spectrum_frequencies.npy",
            energy_quadrature.frequencies,
        )
        np.save(
            output_directory / "detector_energy_spectrum_dU_domega.npy",
            detector_total_energy.spectrum_dU_domega,
        )
    else:
        detector_total_energy = None
        detector_energy_au = factorized_energy.total_energy_au
        energy_method = "factorized-bandwidth"

    aperture_radius = fiber_radius if target_type == "nanofiber" else disk_radius
    incident_energy_au = compute_incident_beam_energy(laser_parameters, pulse_parameters)
    intercepted_incident_energy_au = compute_incident_beam_energy(
        laser_parameters, pulse_parameters, aperture_radius=aperture_radius
    )
    kinetic_balance = compute_electron_kinetic_energy_balance(
        initial_momenta, result.final_momenta, particle_mass=m_e
    )
    larmor_energy_au = float(jnp.sum(result.larmor_energy_per_particle))
    sheet_radius_for_theory = (
        disk_radius if target_type in ("disk", "thin-sheet") else None
    )

    energy_analysis = analyze_energy_transfer(
        detector_energy_au=detector_energy_au,
        incident_energy_au=incident_energy_au,
        doppler_factor=doppler_shift_factor,
        num_electrons=num_electrons,
        detector_energy_factorized_au=factorized_energy.total_energy_au,
        intercepted_incident_energy_au=intercepted_incident_energy_au,
        larmor_energy_au=larmor_energy_au,
        electron_kinetic_energy=kinetic_balance,
        sheet_radius=sheet_radius_for_theory,
        gamma=gamma,
        laser_frequency=laser_frequency,
        physical_num_electrons=physical_electrons,
        physical_bunch_charge_pc=physical_bunch_charge_pc,
    )
    print(
        format_energy_transfer_report(
            energy_analysis,
            detector_spectral_energy=detector_spectral_energy,
            detector_total_energy=detector_total_energy,
            factorized_detector_energy=factorized_energy,
        )
    )
    save_json(
        output_directory / "energy_summary.json",
        build_energy_summary_dict(
            energy_analysis,
            detector_spectral_energy=detector_spectral_energy,
            detector_total_energy=detector_total_energy,
            factorized_detector_energy=factorized_energy,
            frequency_quadrature=energy_quadrature,
            energy_method=energy_method,
        ),
    )


if __name__ == "__main__":
    typer.run(main)
