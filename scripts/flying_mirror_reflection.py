from datetime import UTC, datetime
from math import cos, pi, radians, sin, sqrt
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

from electrodynamics.angular_momentum import (
    compute_detector_oam_spectral_density,
    compute_oam_per_photon,
    compute_particle_angular_momentum_z,
    integrate_detector_oam_spectral_density,
)
from electrodynamics.beams import LaguerreGaussBeamParameters
from electrodynamics.constants import (
    DEFAULT_LASER_FREQUENCY,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    HARTREE_TO_EV,
    SPEED_OF_LIGHT,
    au_length_to_meters,
)
from electrodynamics.detector import (
    build_detector_grid,
    create_detector_parameters,
    project_fields_to_detector_frame,
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
    compute_relativistic_reflection_parameters,
    compute_scattered_electric_and_magnetic_fields,
)
from electrodynamics.initial_conditions import (
    generate_initial_particle_momenta_moving_towards_laser,
    generate_initial_positions_on_thin_sheet,
    to_position_four_vectors,
)
from electrodynamics.integrate import integration_step_rk4_lab_time
from electrodynamics.jax import initialize_jax, iterate_sharded_particle_batches
from electrodynamics.plotting import (
    plot_angular_momentum_distribution,
    plot_detector_radiation_and_oam,
    plot_experimental_setup,
    plot_final_momentum_distribution,
    plot_scattered_radiation_spectrum,
    save_detector_field_component_plots,
    save_figure,
)
from electrodynamics.polarization import get_polarization_by_name
from electrodynamics.pulse import CosineSquaredPulseParameters
from electrodynamics.serialization import prepare_output_directory, save_json

c = SPEED_OF_LIGHT
m_e = ELECTRON_MASS
q = ELECTRON_CHARGE

type IterationState = tuple[
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
def integrate_particles_flying_mirror(
    initial_positions: jax.Array,
    initial_momenta: jax.Array,
    start_time: jdc.Static[float],
    end_time: jdc.Static[float],
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[CosineSquaredPulseParameters],
    incidence_angle: jdc.Static[float],
    central_frequency: jdc.Static[float],
    frequency_width: jdc.Static[float],
    num_frequencies: jdc.Static[int],
    spectrum_measurement_position: jdc.Static[jax.Array],
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
    """Integrates electron trajectories in the thin relativistic flying mirror
    in laboratory time and accumulates the coherently scattered Liénard-Wiechert
    electric and magnetic fields across the frequency spectrum and 2D detector grid.
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

    num_time_steps = round((end_time - start_time) / time_step)

    def scan_fn(u: IterationState, _: None) -> tuple[IterationState, None]:
        (
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

        new_positions, new_momenta = integration_step_rk4_lab_time(
            previous_positions,
            previous_momenta,
            time_step,
            laser_parameters,
            pulse_parameters,
            incidence_angle=incidence_angle,
        )

        def spec_fn(freq: jax.Array) -> tuple[jax.Array, jax.Array]:
            return compute_scattered_electric_and_magnetic_fields(
                freq,
                new_positions,
                new_momenta,
                initial_positions,
                spectrum_measurement_position,
                is_lab_time=True,
                initial_momentum=initial_momenta,
            )

        electric_field_spec, magnetic_field_spec = jax.vmap(
            spec_fn, in_axes=0, out_axes=1
        )(frequencies)

        new_scattered_electric_field = (
            previous_scattered_electric_field + time_step * electric_field_spec
        )
        new_scattered_magnetic_field = (
            previous_scattered_magnetic_field + time_step * magnetic_field_spec
        )

        def det_fn(det_pos: jax.Array) -> tuple[jax.Array, jax.Array]:
            return compute_scattered_electric_and_magnetic_fields(
                central_frequency,
                new_positions,
                new_momenta,
                initial_positions,
                det_pos,
                is_lab_time=True,
                initial_momentum=initial_momenta,
            )

        electric_field_det, magnetic_field_det = jax.vmap(
            det_fn, in_axes=0, out_axes=1
        )(detector_positions)

        new_detector_electric_field = (
            previous_detector_electric_field + time_step * electric_field_det
        )
        new_detector_magnetic_field = (
            previous_detector_magnetic_field + time_step * magnetic_field_det
        )
        new_larmor_energy = (
            previous_larmor_energy
            + compute_larmor_radiated_energy_step(
                new_positions,
                new_momenta,
                time_step,
                laser_parameters,
                pulse_parameters,
                incidence_angle=incidence_angle,
                is_lab_time=True,
            )
        )

        if num_energy_frequencies > 0:
            dE_mf, dB_mf = compute_coherent_detector_fields_multifreq(
                new_positions,
                new_momenta,
                initial_positions,
                detector_positions,
                energy_frequencies,
                is_lab_time=True,
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
class FlyingMirrorResult:
    frequencies: np.ndarray
    final_positions: jax.Array
    final_momenta: jax.Array
    scattered_electric_field_spectrum: jax.Array
    scattered_magnetic_field_spectrum: jax.Array
    detector_electric_field: jax.Array
    detector_magnetic_field: jax.Array
    detector_incoherent_intensity: jax.Array
    larmor_energy_per_particle: jax.Array
    detector_electric_field_multifreq: jax.Array
    detector_magnetic_field_multifreq: jax.Array


def simulate_flying_mirror(
    initial_positions: jax.Array,
    initial_momenta: jax.Array,
    start_time: jdc.Static[float],
    end_time: jdc.Static[float],
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[CosineSquaredPulseParameters],
    incidence_angle: jdc.Static[float],
    central_frequency: jdc.Static[float],
    frequency_width: jdc.Static[float],
    num_frequencies: jdc.Static[int],
    spectrum_measurement_position: jax.Array,
    detector_positions: jdc.Static[jax.Array],
    energy_frequencies: jax.Array,
    batch_size: jdc.Static[int],
) -> FlyingMirrorResult:
    print(
        "[bold blue]Integrating trajectories for thin electron sheet and computing coherent scattered field...[/bold blue]"
    )

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
    detector_incoherent_intensity = jnp.zeros(
        (len(detector_positions),), dtype=jnp.float64
    )
    detector_electric_field_multifreq = jnp.zeros(
        (len(detector_positions), len(energy_frequencies), 3), dtype=jnp.complex128
    )
    detector_magnetic_field_multifreq = jnp.zeros(
        (len(detector_positions), len(energy_frequencies), 3), dtype=jnp.complex128
    )

    for (
        index,
        num_batches,
        batch_positions,
        batch_momenta,
    ) in iterate_sharded_particle_batches(
        initial_positions, initial_momenta, batch_size
    ):
        print(f"Batch #{index + 1}/{num_batches}")
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
        ) = integrate_particles_flying_mirror(
            batch_positions,
            batch_momenta,
            start_time,
            end_time,
            time_step,
            laser_parameters,
            pulse_parameters,
            incidence_angle,
            central_frequency,
            frequency_width,
            num_frequencies,
            spectrum_measurement_position,
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
        detector_incoherent_intensity += jnp.sum(
            jnp.sum(jnp.abs(batch_detector_electric_field) ** 2, axis=-1),
            axis=0,
        ).block_until_ready()
        detector_electric_field_multifreq += (
            batch_detector_electric_field_mf.block_until_ready()
        )
        detector_magnetic_field_multifreq += (
            batch_detector_magnetic_field_mf.block_until_ready()
        )

    return FlyingMirrorResult(
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
        detector_incoherent_intensity,
        jnp.concatenate(larmor_energies),
        detector_electric_field_multifreq,
        detector_magnetic_field_multifreq,
    )


def main(
    radial_index: Annotated[
        int, typer.Option(help="Radial index p of the incident LG vortex beam")
    ] = 0,
    azimuthal_index: Annotated[
        int,
        typer.Option(
            help="Azimuthal index l (topological charge) of the incident LG vortex beam"
        ),
    ] = 1,
    polarization_mode: Annotated[
        Literal["linear-x", "left-circular", "right-circular"],
        typer.Option(
            help="Incident laser polarization: 'linear-x' (LP), 'left-circular' (LCP), or 'right-circular' (RCP)"
        ),
    ] = "linear-x",
    a_0: Annotated[
        float,
        typer.Option(help="Normalized laser amplitude a_0 = e E_0 / (m_e c omega_0)"),
    ] = 0.1,
    waist_radius_lambda: Annotated[
        float,
        typer.Option(
            help="Laser waist radius w_0 in multiples of wavelength lambda_0 (6.25 in Chen et al. 2022)"
        ),
    ] = 6.25,
    pulse_duration_periods: Annotated[
        float,
        typer.Option(
            help="Pulse half-duration tau in laser periods T_0 (10.0 T_0 = 26.7 fs in Chen et al. 2022)"
        ),
    ] = 10.0,
    beta: Annotated[
        float,
        typer.Option(
            help="Relativistic flying mirror velocity v/c (0.5 in Chen et al. 2022; ignored if --gamma > 1.0)"
        ),
    ] = 0.5,
    gamma: Annotated[
        float,
        typer.Option(
            help="Optional mirror Lorentz factor gamma > 1.0 (overrides --beta if > 1.0, e.g. 10.0 or 25.0 for X-ray regime)"
        ),
    ] = 0.0,
    incidence_angle_deg: Annotated[
        float,
        typer.Option(
            help="Laser incidence angle theta_in in degrees relative to the flying mirror normal (0.0 for head-on, 30.0 = pi/6 in Chen et al. 2022)"
        ),
    ] = 0.0,
    sheet_radius_lambda: Annotated[
        float,
        typer.Option(help="Radius of the thin electron sheet in units of lambda_0"),
    ] = 16.0,
    sheet_thickness_lambda: Annotated[
        float,
        typer.Option(
            help="Thickness of the electron sheet in units of lambda_0 (0.0 for ideal monolayer)"
        ),
    ] = 0.0,
    sampling_method: Annotated[
        str,
        typer.Option(
            help="Sampling scheme for electrons on the thin sheet: 'polar-grid', 'fibonacci', or 'random'"
        ),
    ] = "polar-grid",
    num_electrons: Annotated[
        int, typer.Option(help="Number of electrons in the flying mirror sheet")
    ] = 2048,
    batch_size: Annotated[
        int, typer.Option(help="Number of electrons processed per batch")
    ] = 1024,
    detector_distance_lambda: Annotated[
        float,
        typer.Option(
            help="Minimum far-field detector distance from the collision point in units of lambda_0"
        ),
    ] = 5000.0,
    min_far_field_rayleighs: Annotated[
        float,
        typer.Option(
            help="Minimum detector distance as a multiple of the reflected beam Rayleigh range z_{R,1}"
        ),
    ] = 12.0,
    detector_width_beams: Annotated[
        float,
        typer.Option(
            help="Detector width and height as a multiple of the diffracted beam waist w_1(R_det) at the detector"
        ),
    ] = 5.0,
    detector_shape: Annotated[
        Literal["rect", "disk"],
        typer.Option(help="Detector geometry: 'rect' (Cartesian) or 'disk' (polar)"),
    ] = "rect",
    detector_grid_size: Annotated[
        int, typer.Option(help="Detector resolution (grid_size x grid_size pixels for 'rect')")
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
    frequency_window_percentage: Annotated[
        float,
        typer.Option(
            help="Fractional frequency window around central Doppler-shifted frequency"
        ),
    ] = 0.4,
    num_frequencies: Annotated[
        int, typer.Option(help="Number of frequency points in the spectrum")
    ] = 256,
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
    steps_per_period: Annotated[
        int, typer.Option(help="Number of RK4 integration steps per laser period T_0")
    ] = 150,
    exact_paraxial: Annotated[
        bool,
        typer.Option(
            help="Use exact paraxial derivatives for longitudinal EM fields (E_z, B_z)"
        ),
    ] = True,
    seed: Annotated[int, typer.Option(help="Random number generator seed")] = 42,
    run_tag: Annotated[
        str,
        typer.Option(help="Optional custom tag for the output directory under runs/"),
    ] = "",
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Overwrite latest run directory instead of creating a timestamped directory"
        ),
    ] = False,
) -> None:
    """Reproduces the reflection of a Laguerre-Gaussian (LG) vortex laser beam from a
    relativistic flying mirror (a thin sheet of relativistic electrons), following
    Chen et al., Scientific Reports 12:12524 (2022), and computes the coherently
    scattered radiation field and orbital angular momentum on a far-field detector.
    """
    run_datetime = datetime.now(UTC).astimezone()
    print(
        f"[bold green]Started flying mirror reflection study at {run_datetime}[/bold green]"
    )

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
    laser_period = (2 * pi) / laser_frequency

    amplitude = a_0 * m_e * c * laser_frequency / abs(q)
    waist_radius = waist_radius_lambda * laser_wavelength
    polarization, _ = get_polarization_by_name(polarization_mode)

    laser_parameters = LaguerreGaussBeamParameters(
        frequency=laser_frequency,
        wavelength=laser_wavelength,
        amplitude=amplitude,
        polarization=polarization,
        waist_radius=waist_radius,
        radial_index=radial_index,
        azimuthal_index=azimuthal_index,
        exact_paraxial=exact_paraxial,
    )

    if gamma > 1.0:
        beta = sqrt(1.0 - 1.0 / (gamma**2))
    else:
        gamma = 1.0 / sqrt(1.0 - beta**2)
    incidence_angle = radians(incidence_angle_deg)

    theta_in_prime, theta_re, doppler_factor, profile_tilt_angle = (
        compute_relativistic_reflection_parameters(gamma, incidence_angle)
    )

    reflected_frequency = laser_frequency * doppler_factor
    reflected_wavelength = laser_wavelength / doppler_factor
    reflected_wavelength_um = 0.8 / doppler_factor
    reflected_wavelength_nm = 800.0 / doppler_factor
    photon_energy_ev = reflected_frequency * HARTREE_TO_EV

    print(
        f"Flying mirror: beta = {beta:.6f}, gamma = {gamma:.4f}, incidence angle theta_in = {incidence_angle_deg:.1f} deg ({incidence_angle:.4f} rad)"
    )
    print(
        f"Relativistic reflection: theta_re = {theta_re:.5f} rad ({np.degrees(theta_re):.3f} deg), "
        f"Doppler factor = {doppler_factor:.2f}x, reflected wavelength lambda_1 = {reflected_wavelength_nm:.3f} nm "
        f"(hbar*omega_1 = {photon_energy_ev:.2f} eV = {photon_energy_ev / 1000.0:.3f} keV)"
    )

    generator = np.random.default_rng(seed)
    sheet_radius = sheet_radius_lambda * laser_wavelength
    sheet_thickness = sheet_thickness_lambda * laser_wavelength

    # Timing: schedule collision at the origin (0, 0, 0) at t = t_coll so that the
    # entire sheet starts ahead of the pulse at t = 0 and exits behind the pulse by t = 2 * t_coll
    tau_0 = pulse_duration_periods * laser_period
    t_coll = (
        tau_0 + (sheet_radius * abs(sin(incidence_angle))) / c + 1.5 * laser_period
    ) / (1.0 + beta * cos(incidence_angle))
    z_mirror_initial = beta * c * t_coll

    pulse_parameters = CosineSquaredPulseParameters(phi_0=t_coll, tau_0=tau_0)

    initial_positions_3d = generate_initial_positions_on_thin_sheet(
        disk_radius=sheet_radius,
        num_particles=num_electrons,
        z_center=z_mirror_initial,
        thickness=sheet_thickness,
        sampling_method=sampling_method,
        generator=generator,
    )
    initial_positions = to_position_four_vectors(initial_positions_3d)
    initial_momenta = generate_initial_particle_momenta_moving_towards_laser(
        num_particles=num_electrons, gamma=gamma, particle_mass=m_e
    )

    initial_positions_jnp = jnp.asarray(initial_positions)
    initial_momenta_jnp = jnp.asarray(initial_momenta)

    integration_start_time = 0.0
    integration_end_time = 2.0 * t_coll
    time_step = laser_period / steps_per_period

    print(
        f"Integrating from t = 0 to t = {integration_end_time / laser_period:.2f} T_0 "
        f"(collision at t = {t_coll / laser_period:.2f} T_0, z = 0) with dt = T_0 / {steps_per_period}"
    )

    reflected_rayleigh_range = (pi * waist_radius**2) / reflected_wavelength
    detector_distance = max(
        detector_distance_lambda * laser_wavelength,
        min_far_field_rayleighs * reflected_rayleigh_range,
    )
    actual_detector_distance_lambda = detector_distance / laser_wavelength
    diffracted_beam_waist = waist_radius * sqrt(
        1.0 + (detector_distance / reflected_rayleigh_range) ** 2
    )
    detector_width = detector_width_beams * diffracted_beam_waist

    detector_distance_m = au_length_to_meters(detector_distance)
    detector_width_m = au_length_to_meters(detector_width)

    print(
        f"Detector placed in far field at R_det = {actual_detector_distance_lambda:.1f} lambda_0 "
        f"({detector_distance / reflected_rayleigh_range:.1f} x z_{{R,1}}, {detector_distance_m:.4f} m), "
        f"diffracted waist w_1 = {diffracted_beam_waist / laser_wavelength:.1f} lambda_0, "
        f"span = {detector_width_m * 1e3:.3f} mm"
    )

    energy_quadrature = (
        build_frequency_quadrature(
            reflected_frequency,
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

    detector_parameters = create_detector_parameters(
        detector_shape,
        span=detector_width,
        grid_size=detector_grid_size,
        num_radial=detector_num_radial,
        num_azimuthal=detector_num_azimuthal,
        radial_sampling=detector_radial_sampling,
    )
    detector_grid = build_detector_grid(
        detector_parameters,
        distance=detector_distance,
        reflection_angle=theta_re,
        direction=-1,
    )
    detector_positions_flat = detector_grid.lab_positions_flat
    detector_local_coords_2d = detector_grid.local_coords_2d
    detector_area_weights = detector_grid.area_weights_2d
    detector_positions_jnp = jnp.asarray(detector_positions_flat)

    ring_peak_offset = (
        diffracted_beam_waist * sqrt(max(1, abs(azimuthal_index)) / 2.0)
        if azimuthal_index != 0
        else 0.0
    )
    spectrum_measurement_position = jnp.array(
        [
            detector_distance * sin(theta_re) + ring_peak_offset * cos(theta_re),
            0.0,
            -detector_distance * cos(theta_re) + ring_peak_offset * sin(theta_re),
        ]
    )

    frequency_width = frequency_window_percentage * reflected_frequency

    output_directory = prepare_output_directory(
        "flying_mirror",
        run_datetime,
        overwrite=overwrite,
        overwrite_run_tag=f"theta_{round(incidence_angle_deg)}deg_latest",
        run_tag=run_tag,
    )

    configuration = {
        "laser_parameters": laser_parameters,
        "pulse_parameters": pulse_parameters,
        "polarization_mode": polarization_mode,
        "a_0": a_0,
        "beta": beta,
        "gamma": gamma,
        "incidence_angle_deg": incidence_angle_deg,
        "incidence_angle_rad": incidence_angle,
        "theta_in_mirror_frame_rad": theta_in_prime,
        "theta_re_rad": theta_re,
        "doppler_factor": doppler_factor,
        "reflected_frequency_au": reflected_frequency,
        "reflected_wavelength_um": reflected_wavelength_um,
        "profile_tilt_angle_rad": profile_tilt_angle,
        "sheet_radius_lambda": sheet_radius_lambda,
        "sheet_thickness_lambda": sheet_thickness_lambda,
        "sampling_method": sampling_method,
        "num_electrons": num_electrons,
        "detector_distance_lambda": detector_distance_lambda,
        "actual_detector_distance_lambda": actual_detector_distance_lambda,
        "detector_distance_m": detector_distance_m,
        "diffracted_beam_waist_lambda": diffracted_beam_waist / laser_wavelength,
        "detector_width_lambda": detector_width / laser_wavelength,
        "detector_width_m": detector_width_m,
        "detector_shape": detector_shape,
        "detector_parameters": detector_parameters,
        "energy_num_frequencies": energy_num_frequencies,
        "energy_bandwidth_fraction": energy_bandwidth_fraction,
        "physical_electrons": physical_electrons,
        "physical_bunch_charge_pc": physical_bunch_charge_pc,
        "time_start": integration_start_time,
        "time_end": integration_end_time,
        "time_delta": time_step,
        "spectrum_measurement_position": spectrum_measurement_position,
    }
    save_json(output_directory / "configuration.json", configuration)

    print("Plotting experimental setup")
    fig = plt.figure(figsize=(8, 5), dpi=300)
    plot_experimental_setup(
        fig,
        np.asarray(initial_positions),
        np.asarray(initial_momenta),
        detector_parameters,
        np.asarray(detector_positions_flat),
    )
    save_figure(fig, output_directory / "experimental_setup")

    start_wall = perf_counter()
    result = simulate_flying_mirror(
        initial_positions_jnp,
        initial_momenta_jnp,
        integration_start_time,
        integration_end_time,
        time_step,
        laser_parameters,
        pulse_parameters,
        incidence_angle,
        reflected_frequency,
        frequency_width,
        num_frequencies,
        spectrum_measurement_position,
        detector_positions_jnp,
        energy_frequencies_jnp,
        batch_size,
    )
    duration = perf_counter() - start_wall
    print(f"[bold green]Simulation completed in {duration:.2f} seconds![/bold green]")

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

    spec_E_np = np.asarray(result.scattered_electric_field_spectrum)
    spec_norms = np.linalg.vector_norm(spec_E_np, axis=-1)
    peak_freq_idx = int(np.argmax(spec_norms))
    simulated_peak_frequency = float(result.frequencies[peak_freq_idx])
    simulated_wavelength_um = 0.8 * (laser_frequency / simulated_peak_frequency)

    fig = plt.figure(dpi=200)
    plot_scattered_radiation_spectrum(
        fig,
        reflected_frequency,
        result.frequencies,
        spec_E_np,
    )
    save_figure(
        fig, output_directory / "electric_field_spectrum", formats=("pdf", "png")
    )

    det_E_lab_2d = detector_parameters.reshape_to_grid(
        np.asarray(result.detector_electric_field)
    )
    det_B_lab_2d = detector_parameters.reshape_to_grid(
        np.asarray(result.detector_magnetic_field)
    )
    det_E_local, det_B_local = project_fields_to_detector_frame(
        det_E_lab_2d, det_B_lab_2d, reflection_angle=theta_re
    )

    dL_z = compute_detector_oam_spectral_density(
        detector_grid, det_E_local, reflected_frequency
    )
    total_radiated_Lz = integrate_detector_oam_spectral_density(
        detector_local_coords_2d, dL_z, area_weights=detector_area_weights
    )
    oam_per_photon_lab = compute_oam_per_photon(
        detector_local_coords_2d,
        det_E_local,
        dL_z,
        reflected_frequency,
        propagation_sign=1,
        area_weights=detector_area_weights,
    )
    oam_per_photon_beam = compute_oam_per_photon(
        detector_local_coords_2d,
        det_E_local,
        dL_z,
        reflected_frequency,
        propagation_sign=-1,
        area_weights=detector_area_weights,
    )

    coherent_intensity_2d = np.sum(np.abs(det_E_local) ** 2, axis=-1)
    incoherent_intensity_2d = detector_parameters.reshape_to_grid(
        np.asarray(result.detector_incoherent_intensity)
    )
    peak_coherent_intensity = float(np.max(coherent_intensity_2d))
    total_detector_area = float(np.sum(detector_area_weights))
    mean_incoherent_intensity = (
        float(np.sum(incoherent_intensity_2d * detector_area_weights) / total_detector_area)
        if total_detector_area > 0.0
        else float(np.mean(incoherent_intensity_2d))
    )
    coherence_enhancement_factor = (
        peak_coherent_intensity / mean_incoherent_intensity
        if mean_incoherent_intensity > 0
        else 0.0
    )

    print(
        f"Peak coherent intensity |E_coh|^2 = {peak_coherent_intensity:.4e}, "
        f"mean incoherent intensity sum(|E_i|^2) = {mean_incoherent_intensity:.4e} "
        f"=> Coherence enhancement factor = {coherence_enhancement_factor:.1f}x"
    )
    print(
        f"Total radiated OAM spectral density on detector dL_z^EM/domega = {total_radiated_Lz:.4e} a.u. "
        f"(OAM per photon: lab L_z = {oam_per_photon_lab:+.4f}, beam-axis = {oam_per_photon_beam:+.4f})"
    )

    reflected_wavenumber = (2 * pi) / reflected_wavelength
    if photon_energy_ev >= 100.0:
        energy_label = f"\\hbar\\omega_1={photon_energy_ev / 1000.0:.2f}\\,\\mathrm{{keV}}, \\lambda_1={reflected_wavelength_nm:.2f}\\,\\mathrm{{nm}}"
    else:
        energy_label = f"\\lambda_1={reflected_wavelength_um:.3f}\\,\\mu\\mathrm{{m}}"
    fig = plt.figure(dpi=200)
    plot_detector_radiation_and_oam(
        fig,
        detector_local_coords_2d,
        det_E_local,
        dL_z,
        laser_wavelength,
        title=(
            f"Flying Mirror Reflected Vortex Beam ($\\gamma={gamma:.2f}, \\theta_{{\\mathrm{{in}}}}={incidence_angle_deg:.0f}^\\circ \\to "
            f"{energy_label}$)"
        ),
        wave_number=reflected_wavenumber,
        area_weights=detector_area_weights,
        detector_parameters=detector_parameters,
    )
    save_figure(
        fig, output_directory / "detector_radiation_and_oam", formats=("pdf", "png")
    )

    save_detector_field_component_plots(
        output_directory,
        det_E_local,
        detector_parameters,
        waist_radius,
        formats=("pdf", "png"),
        local_coords_2d=detector_local_coords_2d,
    )

    np.save(output_directory / "detector_local_coords.npy", detector_local_coords_2d)
    np.save(output_directory / "detector_area_weights.npy", detector_area_weights)
    np.save(output_directory / "detector_electric_field.npy", det_E_local)
    np.save(output_directory / "detector_magnetic_field.npy", det_B_local)
    np.save(output_directory / "detector_dL_z.npy", dL_z)
    np.save(output_directory / "spectrum_frequencies.npy", result.frequencies)
    np.save(output_directory / "spectrum_electric_field.npy", spec_E_np)

    detector_spectral_energy = compute_detector_spectral_energy(
        det_E_local,
        det_B_local,
        detector_area_weights,
        normal_vector=detector_grid.normal_local,
        detector_positions=detector_local_coords_2d,
    )
    factorized_energy = compute_factorized_detector_energy(
        det_E_local,
        det_B_local,
        detector_area_weights,
        result.frequencies,
        spec_E_np,
        reference_frequency=reflected_frequency,
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
            detector_area_weights,
            energy_quadrature.weights,
            normal_vector=detector_grid.normal_lab,
            detector_positions=detector_grid.reshape_to_grid(detector_positions_flat),
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

    incident_energy_au = compute_incident_beam_energy(laser_parameters, pulse_parameters)
    intercepted_incident_energy_au = compute_incident_beam_energy(
        laser_parameters,
        pulse_parameters,
        aperture_radius=sheet_radius,
        incidence_angle=incidence_angle,
    )
    kinetic_balance = compute_electron_kinetic_energy_balance(
        initial_momenta, result.final_momenta, particle_mass=m_e
    )
    larmor_energy_au = float(jnp.sum(result.larmor_energy_per_particle))

    energy_analysis = analyze_energy_transfer(
        detector_energy_au=detector_energy_au,
        incident_energy_au=incident_energy_au,
        doppler_factor=doppler_factor,
        num_electrons=num_electrons,
        detector_energy_factorized_au=factorized_energy.total_energy_au,
        intercepted_incident_energy_au=intercepted_incident_energy_au,
        larmor_energy_au=larmor_energy_au,
        electron_kinetic_energy=kinetic_balance,
        sheet_radius=sheet_radius,
        gamma=gamma,
        laser_frequency=laser_frequency,
        incidence_angle=incidence_angle,
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

    summary = {
        "radial_index": radial_index,
        "azimuthal_index": azimuthal_index,
        "polarization_mode": polarization_mode,
        "a_0": a_0,
        "beta": beta,
        "gamma": gamma,
        "incidence_angle_deg": incidence_angle_deg,
        "incidence_angle_rad": incidence_angle,
        "theoretical_theta_re_rad": theta_re,
        "theoretical_doppler_factor": doppler_factor,
        "theoretical_reflected_frequency_au": reflected_frequency,
        "simulated_peak_frequency_au": simulated_peak_frequency,
        "theoretical_reflected_wavelength_um": reflected_wavelength_um,
        "theoretical_reflected_wavelength_nm": reflected_wavelength_nm,
        "theoretical_photon_energy_ev": photon_energy_ev,
        "theoretical_photon_energy_kev": photon_energy_ev / 1000.0,
        "simulated_peak_wavelength_um": simulated_wavelength_um,
        "simulated_peak_wavelength_nm": simulated_wavelength_um * 1000.0,
        "simulated_photon_energy_ev": simulated_peak_frequency * HARTREE_TO_EV,
        "simulated_photon_energy_kev": (simulated_peak_frequency * HARTREE_TO_EV)
        / 1000.0,
        "actual_detector_distance_lambda": actual_detector_distance_lambda,
        "detector_distance_m": detector_distance_m,
        "detector_width_m": detector_width_m,
        "theoretical_profile_tilt_angle_rad": profile_tilt_angle,
        "peak_coherent_intensity": peak_coherent_intensity,
        "mean_incoherent_intensity": mean_incoherent_intensity,
        "coherence_enhancement_factor": coherence_enhancement_factor,
        "total_radiated_Lz_au": total_radiated_Lz,
        "oam_per_photon_lab": oam_per_photon_lab,
        "oam_per_photon_beam": oam_per_photon_beam,
        "energy_method": energy_method,
        "detector_energy_au": energy_analysis.detector_energy_au,
        "detector_energy_joules": energy_analysis.detector_energy_joules,
        "detector_energy_factorized_au": factorized_energy.total_energy_au,
        "incident_energy_au": energy_analysis.incident_energy_au,
        "larmor_energy_au": larmor_energy_au,
        "laser_energy_fraction": energy_analysis.laser_energy_fraction,
        "photon_reflectivity": energy_analysis.photon_reflectivity,
        "thin_sheet_theory_energy_au": (
            energy_analysis.thin_sheet_benchmark.weak_theory_energy_au
            if energy_analysis.thin_sheet_benchmark is not None
            else None
        ),
        "detector_shape": detector_shape,
        "num_electrons": num_electrons,
        "sampling_method": sampling_method,
    }
    save_json(output_directory / "flying_mirror_summary.json", summary)

    print(f"[bold green]Saved all plots and data to {output_directory}[/bold green]")


if __name__ == "__main__":
    typer.run(main)
