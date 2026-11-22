from datetime import UTC, datetime
from math import pi
from time import perf_counter
from typing import Annotated

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import typer
from rich import print

from electrodynamics.angular_momentum import (
    compute_angular_momentum_derivative,
    compute_particle_angular_momentum_z,
)
from electrodynamics.beams import (
    GaussianBeamParameters,
    LaguerreGaussBeamParameters,
    LaserParameters,
)
from electrodynamics.constants import (
    DEFAULT_LASER_FREQUENCY,
    HARTREE_TO_EV,
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
    DetectorParameters,
    initialize_detector_positions_negative_z,
)
from electrodynamics.fields import compute_coherent_detector_and_spectrum_fields
from electrodynamics.initial_conditions import (
    generate_initial_particle_momenta_at_rest,
    generate_nanofiber_initial_positions,
    to_position_four_vectors,
)
from electrodynamics.integrate import (
    SnapshotData,
    integration_step_rk4_lab_time_two_pulses,
    parse_snapshot_schedule,
)
from electrodynamics.jax import initialize_jax
from electrodynamics.plotting import (
    plot_detector_radiation_and_oam,
    plot_scattered_radiation_spectrum,
    save_figure,
    save_nanofiber_final_plots,
    save_nanofiber_snapshot_plots,
)
from electrodynamics.polarization import (
    get_polarization_by_name,
    resolve_circular_polarization,
)
from electrodynamics.pulse import (
    CosineSquaredPulseParameters,
    GaussianPulseParameters,
)
from electrodynamics.serialization import prepare_output_directory, save_json


@jdc.jit
def integrate_interval_two_stage(
    initial_positions: jax.Array,
    initial_momenta: jax.Array,
    origin_positions: jax.Array,
    detector_positions: jax.Array,
    initial_detector_E: jax.Array,
    initial_detector_B: jax.Array,
    radiation_frequency: float,
    spectrum_position: jax.Array,
    frequencies: jax.Array,
    initial_spectrum_E: jax.Array,
    num_steps: jdc.Static[int],
    dt: jdc.Static[float],
    drive_laser: jdc.Static[LaguerreGaussBeamParameters],
    drive_pulse: jdc.Static[CosineSquaredPulseParameters],
    probe_laser: jdc.Static[
        LaserParameters | GaussianBeamParameters | LaguerreGaussBeamParameters | None
    ],
    probe_pulse: jdc.Static[GaussianPulseParameters | None],
    probe_direction: jdc.Static[int],
    probe_reference_z: jdc.Static[float],
    radiation_start_time: jdc.Static[float],
    compute_radiation: jdc.Static[bool] = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Integrates particle trajectories over an interval of laboratory time steps under
    both primary drive LG pulse (+z) and counter-propagating probe pulse (-z),
    accumulating high-frequency Doppler-upshifted coherent radiation on the detector.
    """

    def scan_fn(carry, _):
        pos, mom, det_E, det_B, spec_E = carry
        next_pos, next_mom = integration_step_rk4_lab_time_two_pulses(
            pos,
            mom,
            dt,
            drive_laser,
            drive_pulse,
            probe_laser,
            probe_pulse,
            probe_direction=probe_direction,
            probe_reference_z=probe_reference_z,
        )

        lz_step = jnp.sum(compute_particle_angular_momentum_z(next_pos, next_mom))

        if compute_radiation:
            current_time = next_pos[0, 0] / c
            in_window = current_time >= radiation_start_time

            def compute_rad():
                return compute_coherent_detector_and_spectrum_fields(
                    next_pos,
                    next_mom,
                    origin_positions,
                    detector_positions,
                    radiation_frequency,
                    spectrum_position,
                    frequencies,
                    is_lab_time=True,
                )

            def zero_rad():
                return (
                    jnp.zeros_like(det_E),
                    jnp.zeros_like(det_B),
                    jnp.zeros_like(spec_E),
                )

            dE, dB, dE_s = jax.lax.cond(in_window, compute_rad, zero_rad)
            next_det_E = det_E + dt * dE
            next_det_B = det_B + dt * dB
            next_spec_E = spec_E + dt * dE_s
        else:
            next_det_E = det_E
            next_det_B = det_B
            next_spec_E = spec_E

        return (next_pos, next_mom, next_det_E, next_det_B, next_spec_E), lz_step

    (final_pos, final_mom, final_det_E, final_det_B, final_spec_E), lz_history = (
        jax.lax.scan(
            scan_fn,
            (
                initial_positions,
                initial_momenta,
                initial_detector_E,
                initial_detector_B,
                initial_spectrum_E,
            ),
            None,
            length=num_steps,
        )
    )
    return (
        final_pos,
        final_mom,
        final_det_E,
        final_det_B,
        final_spec_E,
        lz_history,
    )


def main(
    # Stage 1: Drive Laser Parameters
    a_0: Annotated[
        float, typer.Option(help="Drive LG laser normalized amplitude parameter")
    ] = 30.0,
    radial_index: Annotated[
        int, typer.Option(help="Radial mode index p of the drive LG beam")
    ] = 0,
    azimuthal_index: Annotated[
        int, typer.Option(help="Azimuthal mode index l of the drive LG beam")
    ] = 1,
    polarization_handedness: Annotated[
        str,
        typer.Option(
            help="Drive beam polarization handedness relative to l: 'same' (helical bunches j_z = l + sigma != 0) or 'opposite' (axisymmetric disks j_z = 0)"
        ),
    ] = "same",
    waist_radius_lambda: Annotated[
        float,
        typer.Option(help="Drive beam waist radius w_0 in laser wavelengths lambda_0"),
    ] = 3.0,
    pulse_duration_periods: Annotated[
        float,
        typer.Option(help="Drive pulse duration parameter tau in laser periods T_0"),
    ] = 5.0,
    # Nanofiber target parameters
    fiber_radius_lambda: Annotated[
        float, typer.Option(help="Nanofiber radius R_0 in laser wavelengths lambda_0")
    ] = 0.5,
    fiber_length_lambda: Annotated[
        float, typer.Option(help="Nanofiber length L_0 in laser wavelengths lambda_0")
    ] = 15.0,
    skin_depth_lambda: Annotated[
        float,
        typer.Option(
            help="If > 0, restrict initial electrons to a skin layer of depth skin_depth_lambda * lambda_0"
        ),
    ] = 0.0,
    num_electrons: Annotated[
        int, typer.Option(help="Number of simulated electrons")
    ] = 2048,
    # Stage 2: Counter-propagating Probe Laser Parameters
    probe_a_0: Annotated[
        float, typer.Option(help="Probe laser normalized amplitude parameter a_probe")
    ] = 0.5,
    probe_laser_type: Annotated[
        str,
        typer.Option(
            help="Probe laser beam profile: 'gaussian', 'plane-wave', or 'laguerre-gauss'"
        ),
    ] = "laguerre-gauss",
    probe_waist_lambda: Annotated[
        float,
        typer.Option(help="Probe laser waist radius in laser wavelengths lambda_0"),
    ] = 5.0,
    probe_wavelength_ratio: Annotated[
        float,
        typer.Option(
            help="Probe wavelength as a multiple of drive wavelength lambda_0"
        ),
    ] = 1.0,
    probe_polarization: Annotated[
        str,
        typer.Option(
            help="Probe laser polarization: 'linear-x', 'linear-y', 'circular'"
        ),
    ] = "linear-x",
    collision_time_periods: Annotated[
        float,
        typer.Option(
            help="Arrival time of probe pulse peak at collision point in units of T_0"
        ),
    ] = 27.0,
    collision_z_lambda: Annotated[
        float,
        typer.Option(
            help="Longitudinal collision coordinate z_coll in units of lambda_0"
        ),
    ] = 25.0,
    probe_duration_periods: Annotated[
        float, typer.Option(help="Probe pulse duration tau_probe in units of T_0")
    ] = 4.0,
    # Runtime and simulation schedule
    t_final_periods: Annotated[
        float, typer.Option(help="Final simulation time in units of laser periods T_0")
    ] = 35.0,
    snapshot_periods: Annotated[
        str,
        typer.Option(
            help="Comma-separated timestamps in T_0 at which to record bunch snapshots (e.g. '10.0,20.0,27.0,32.0')"
        ),
    ] = "10.0,20.0,27.0,32.0",
    steps_per_period: Annotated[
        int, typer.Option(help="Number of time integration steps per laser period T_0")
    ] = 100,
    exact_paraxial: Annotated[
        bool,
        typer.Option(
            help="Use exact paraxial analytical derivatives for longitudinal EM fields"
        ),
    ] = True,
    # High-frequency Doppler-upshifted radiation detection
    compute_radiation: Annotated[
        bool,
        typer.Option(help="Accumulate high-frequency scattered radiation on detector"),
    ] = True,
    radiation_frequency_ratio: Annotated[
        float,
        typer.Option(
            help="High-frequency detection frequency as multiple of omega_probe (e.g. 2500.0). If 0.0, auto-estimate as 4 * gamma_bunch^2"
        ),
    ] = 2500.0,
    radiation_start_period: Annotated[
        float,
        typer.Option(
            help="Laser period timestamp after which to accumulate radiation (e.g. 20.0 T_0)"
        ),
    ] = 20.0,
    detector_distance_lambda: Annotated[
        float,
        typer.Option(
            help="Detector distance z_det from entrance face in laser wavelengths lambda_0"
        ),
    ] = 80.0,
    detector_width_lambda: Annotated[
        float,
        typer.Option(
            help="Detector transverse width and height in laser wavelengths lambda_0"
        ),
    ] = 10.0,
    detector_grid_size: Annotated[
        int, typer.Option(help="Detector pixel grid resolution (grid_size x grid_size)")
    ] = 32,
    spectrum_num_frequencies: Annotated[
        int,
        typer.Option(help="Number of frequency sample points for radiation spectrum"),
    ] = 64,
    spectrum_window_fraction: Annotated[
        float,
        typer.Option(
            help="Fractional bandwidth window around central high frequency for spectrum"
        ),
    ] = 0.8,
    seed: Annotated[int, typer.Option(help="Random number generator seed")] = 42,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Overwrite last run directory instead of creating a timestamped run directory"
        ),
    ] = False,
) -> None:
    """Simulates the two-stage all-optical coherent high-frequency (X-ray / gamma-ray) source:
    Stage 1: Intense co-propagating LG laser pulse drives helical attosecond bunches from a nanofiber.
    Stage 2: Counter-propagating probe laser pulse collides head-on with relativistic bunches,
             generating Doppler-upshifted (omega ~ 4 * gamma^2 * omega_0) coherent vortex radiation.
    """
    run_datetime = datetime.now(UTC).astimezone()
    print(
        f"[bold green]Starting Two-Stage Nanofiber Bunching + Probe Collision Simulation at {run_datetime}[/bold green]"
    )

    initialize_jax()

    laser_frequency = DEFAULT_LASER_FREQUENCY
    laser_wavelength = (2 * pi * c) / laser_frequency
    laser_period = (2 * pi) / laser_frequency

    amplitude_drive = a_0 * m_e * c * laser_frequency / abs(q)
    waist_drive = waist_radius_lambda * laser_wavelength
    fiber_radius = fiber_radius_lambda * laser_wavelength
    fiber_length = fiber_length_lambda * laser_wavelength

    drive_pol, sigma = resolve_circular_polarization(
        azimuthal_index, polarization_handedness
    )

    j_z = azimuthal_index + sigma
    print(
        f"Drive beam mode: l = {azimuthal_index}, sigma = {sigma} => total angular momentum j_z = {j_z}"
    )
    if j_z == 0:
        print(
            "[cyan]j_z = 0: Spin-orbit cancellation gives cylindrically symmetric force (axisymmetric disk bunches).[/cyan]"
        )
    else:
        print(
            f"[magenta]j_z = {j_z} != 0: Helical force creates |j_z| = {abs(j_z)} intertwined spiral bunches carrying OAM.[/magenta]"
        )

    drive_laser = LaguerreGaussBeamParameters(
        frequency=laser_frequency,
        wavelength=laser_wavelength,
        amplitude=amplitude_drive,
        polarization=drive_pol,
        waist_radius=waist_drive,
        radial_index=radial_index,
        azimuthal_index=azimuthal_index,
        exact_paraxial=exact_paraxial,
    )
    tau_drive = pulse_duration_periods * laser_period
    drive_pulse = CosineSquaredPulseParameters(phi_0=-tau_drive, tau_0=tau_drive)

    probe_freq = laser_frequency / probe_wavelength_ratio
    probe_wl = laser_wavelength * probe_wavelength_ratio
    amplitude_probe = probe_a_0 * m_e * c * probe_freq / abs(q)
    waist_probe = probe_waist_lambda * laser_wavelength

    probe_pol, _ = get_polarization_by_name(probe_polarization)

    if probe_laser_type == "gaussian":
        probe_laser = GaussianBeamParameters(
            frequency=probe_freq,
            wavelength=probe_wl,
            amplitude=amplitude_probe,
            polarization=probe_pol,
            waist_radius=waist_probe,
            exact_paraxial=exact_paraxial,
        )
    elif probe_laser_type == "laguerre-gauss":
        probe_laser = LaguerreGaussBeamParameters(
            frequency=probe_freq,
            wavelength=probe_wl,
            amplitude=amplitude_probe,
            polarization=probe_pol,
            waist_radius=waist_probe,
            radial_index=0,
            azimuthal_index=1,
            exact_paraxial=exact_paraxial,
        )
    else:
        probe_laser = LaserParameters(
            frequency=probe_freq,
            wavelength=probe_wl,
            amplitude=amplitude_probe,
            polarization=probe_pol,
        )

    t_coll = collision_time_periods * laser_period
    z_coll = collision_z_lambda * laser_wavelength
    tau_probe = probe_duration_periods * laser_period
    probe_pulse = GaussianPulseParameters(phi_0=t_coll, tau_0=tau_probe)

    print(
        f"Probe beam: a_probe = {probe_a_0}, waist = {probe_waist_lambda} lambda_0, counter-propagating towards -z"
    )
    print(
        f"Collision designed at: t = {collision_time_periods:.1f} T_0, z = {collision_z_lambda:.1f} lambda_0"
    )

    generator = np.random.default_rng(seed)
    initial_positions_3d = generate_nanofiber_initial_positions(
        generator,
        fiber_radius,
        fiber_length,
        num_electrons,
        skin_depth=skin_depth_lambda * laser_wavelength,
        z_start=0.0,
    )
    initial_positions = to_position_four_vectors(initial_positions_3d)
    initial_momenta = generate_initial_particle_momenta_at_rest(num_electrons, m_e)

    dt = laser_period / steps_per_period
    print(f"Time step dt = {dt:.4f} a.u. ({steps_per_period} steps / T_0)")

    omega_high = radiation_frequency_ratio * probe_freq
    photon_energy_high_kev = (omega_high * HARTREE_TO_EV) / 1000.0
    print(
        f"High-frequency detection frequency: omega = {omega_high:.3f} a.u. ({radiation_frequency_ratio:.1f} x omega_probe)"
    )
    print(f"Upshifted photon energy: {photon_energy_high_kev:.2f} keV (X-ray regime)")

    detector_z = detector_distance_lambda * laser_wavelength
    det_params = DetectorParameters(
        width=detector_width_lambda * laser_wavelength,
        height=detector_width_lambda * laser_wavelength,
        grid_size_x=detector_grid_size,
        grid_size_y=detector_grid_size,
    )
    det_positions_flat = initialize_detector_positions_negative_z(
        det_params, z_distance=detector_z
    )
    det_positions_jnp = jnp.asarray(det_positions_flat)

    spectrum_pos = jnp.array([0.5 * laser_wavelength, 0.0, detector_z])
    freq_min = max(0.1 * omega_high, omega_high * (1.0 - spectrum_window_fraction / 2))
    freq_max = omega_high * (1.0 + spectrum_window_fraction / 2)
    frequencies_np = np.linspace(freq_min, freq_max, spectrum_num_frequencies)
    frequencies_jnp = jnp.asarray(frequencies_np)

    num_det_pixels = detector_grid_size * detector_grid_size
    current_detector_E = jnp.zeros((num_det_pixels, 3), dtype=jnp.complex128)
    current_detector_B = jnp.zeros((num_det_pixels, 3), dtype=jnp.complex128)
    current_spectrum_E = jnp.zeros((spectrum_num_frequencies, 3), dtype=jnp.complex128)

    rad_start_time = radiation_start_period * laser_period
    parsed_snapshots, schedule_times = parse_snapshot_schedule(
        snapshot_periods, laser_period, t_final_periods
    )

    output_directory = prepare_output_directory(
        "two_stage_probe_collision",
        run_datetime,
        overwrite=overwrite,
        overwrite_suffix=f"jz_{j_z}",
        timestamp_suffix=f"jz{j_z}",
    )

    configuration = {
        "drive_laser_a_0": a_0,
        "azimuthal_index": azimuthal_index,
        "j_z": j_z,
        "polarization_handedness": polarization_handedness,
        "probe_a_0": probe_a_0,
        "probe_laser_type": probe_laser_type,
        "collision_time_periods": collision_time_periods,
        "collision_z_lambda": collision_z_lambda,
        "radiation_frequency_ratio": radiation_frequency_ratio,
        "omega_high_au": omega_high,
        "photon_energy_high_kev": photon_energy_high_kev,
        "detector_distance_lambda": detector_distance_lambda,
        "detector_width_lambda": detector_width_lambda,
        "detector_grid_size": detector_grid_size,
        "num_electrons": num_electrons,
        "t_final_periods": t_final_periods,
    }
    save_json(output_directory / "configuration.json", configuration)

    print(
        f"[bold blue]Starting two-stage integration from t = 0 to t = {t_final_periods:.1f} T_0...[/bold blue]"
    )
    sim_start = perf_counter()

    current_positions = jnp.asarray(initial_positions)
    current_momenta = jnp.asarray(initial_momenta)
    initial_positions_jnp = jnp.asarray(initial_positions)
    current_time = 0.0

    recorded_snapshots: list[SnapshotData] = []
    all_lz_history: list[np.ndarray] = []

    for target_time in schedule_times:
        time_to_advance = target_time - current_time
        num_steps = round(time_to_advance / dt)
        if num_steps <= 0:
            continue

        target_period = target_time / laser_period
        print(
            f"Advancing from t = {current_time / laser_period:.1f} T_0 to t = {target_period:.1f} T_0 ({num_steps} steps)..."
        )

        (
            current_positions,
            current_momenta,
            current_detector_E,
            current_detector_B,
            current_spectrum_E,
            lz_seg,
        ) = integrate_interval_two_stage(
            current_positions,
            current_momenta,
            initial_positions_jnp,
            det_positions_jnp,
            current_detector_E,
            current_detector_B,
            omega_high,
            spectrum_pos,
            frequencies_jnp,
            current_spectrum_E,
            num_steps,
            dt,
            drive_laser,
            drive_pulse,
            probe_laser,
            probe_pulse,
            probe_direction=-1,
            probe_reference_z=z_coll,
            radiation_start_time=rad_start_time,
            compute_radiation=compute_radiation,
        )
        current_positions.block_until_ready()
        all_lz_history.append(np.asarray(lz_seg))
        current_time = target_time

        if any(abs(target_period - s) < 0.05 for s in parsed_snapshots):
            recorded_snapshots.append(
                SnapshotData(
                    time=target_period,
                    positions=current_positions,
                    momenta=current_momenta,
                )
            )

    sim_end = perf_counter()
    print(
        f"[bold green]Two-stage simulation completed in {sim_end - sim_start:.2f} seconds![/bold green]"
    )

    print("Generating bunch snapshot plots...")
    save_nanofiber_snapshot_plots(
        output_directory,
        recorded_snapshots,
        laser_wavelength,
        fiber_radius,
        fiber_length,
        waist_drive,
        j_z,
    )

    save_nanofiber_final_plots(
        output_directory,
        np.asarray(current_positions),
        np.asarray(current_momenta),
        all_lz_history,
        laser_wavelength,
        laser_period,
        t_final_periods,
        j_z,
    )

    if compute_radiation:
        print(
            "[bold blue]Analyzing high-frequency Doppler-upshifted radiation on detector...[/bold blue]"
        )
        current_detector_E.block_until_ready()
        det_pos_2d = det_params.reshape_to_grid(np.asarray(det_positions_flat))
        det_E_2d = det_params.reshape_to_grid(np.asarray(current_detector_E))
        det_B_2d = det_params.reshape_to_grid(np.asarray(current_detector_B))

        dL_z = compute_angular_momentum_derivative(det_pos_2d, det_E_2d, omega_high)
        total_radiated_Lz = float(np.sum(np.real(dL_z)))
        print(
            f"Total high-frequency radiated OAM L_z^EM: {total_radiated_Lz:.6e} a.u."
        )
        print(
            f"Max radiation |E_det|: {np.max(np.linalg.norm(det_E_2d, axis=-1)):.6e}"
        )

        fig = plt.figure(dpi=200)
        plot_detector_radiation_and_oam(
            fig,
            det_pos_2d,
            det_E_2d,
            dL_z,
            laser_wavelength,
            title=f"X-ray radiation at $\\hbar\\omega = {photon_energy_high_kev:.2f}\\text{{ keV}}$ ($j_z = {j_z}$)",
        )
        save_figure(
            fig,
            output_directory / "high_freq_detector_radiation_and_oam",
            formats=("png", "pdf"),
        )

        spec_E_np = np.asarray(current_spectrum_E)
        fig = plt.figure(dpi=200)
        plot_scattered_radiation_spectrum(fig, omega_high, frequencies_np, spec_E_np)
        save_figure(
            fig,
            output_directory / "high_freq_radiation_spectrum",
            formats=("pdf", "png"),
        )

        np.save(output_directory / "detector_positions.npy", det_pos_2d)
        np.save(output_directory / "detector_electric_field.npy", det_E_2d)
        np.save(output_directory / "detector_magnetic_field.npy", det_B_2d)
        np.save(output_directory / "detector_dL_z.npy", dL_z)
        np.save(output_directory / "spectrum_frequencies.npy", frequencies_np)
        np.save(output_directory / "spectrum_electric_field.npy", spec_E_np)

        summary = {
            "drive_a_0": a_0,
            "probe_a_0": probe_a_0,
            "j_z": j_z,
            "radiation_frequency_ratio": radiation_frequency_ratio,
            "omega_high_au": omega_high,
            "photon_energy_high_kev": photon_energy_high_kev,
            "total_radiated_Lz": total_radiated_Lz,
            "max_abs_dL_z": float(np.max(np.abs(dL_z))),
            "max_E_det": float(np.max(np.linalg.norm(det_E_2d, axis=-1))),
            "mean_E_det": float(np.mean(np.linalg.norm(det_E_2d, axis=-1))),
            "detector_distance_lambda": detector_distance_lambda,
            "detector_width_lambda": detector_width_lambda,
            "collision_time_periods": collision_time_periods,
            "collision_z_lambda": collision_z_lambda,
        }
        save_json(output_directory / "two_stage_summary.json", summary)


if __name__ == "__main__":
    typer.run(main)
