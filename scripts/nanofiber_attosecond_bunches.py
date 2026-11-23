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
    integrate_detector_oam_spectral_density,
)
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
    integration_step_rk4_lab_time,
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
from electrodynamics.pulse import CosineSquaredPulseParameters
from electrodynamics.serialization import prepare_output_directory, save_json


@jdc.jit
def integrate_interval(
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
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[CosineSquaredPulseParameters],
    compute_radiation: jdc.Static[bool] = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Integrates particle trajectories over an interval of laboratory time steps,
    recording the total beam angular momentum at each step and accumulating
    coherent scattered radiation across the detector grid and spectrum observer.
    """

    def scan_fn(carry, _):
        pos, mom, det_E, det_B, spec_E = carry
        next_pos, next_mom = integration_step_rk4_lab_time(
            pos, mom, dt, laser_parameters, pulse_parameters
        )
        lz_step = jnp.sum(compute_particle_angular_momentum_z(next_pos, next_mom))

        if compute_radiation:
            dE, dB, dE_spec = compute_coherent_detector_and_spectrum_fields(
                next_pos,
                next_mom,
                origin_positions,
                detector_positions,
                radiation_frequency,
                spectrum_position,
                frequencies,
                is_lab_time=True,
            )
            next_det_E = det_E + dt * dE
            next_det_B = det_B + dt * dB
            next_spec_E = spec_E + dt * dE_spec
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
    a_0: Annotated[
        float, typer.Option(help="Normalized laser field amplitude parameter")
    ] = 30.0,
    radial_index: Annotated[
        int, typer.Option(help="Radial mode index p of the LG beam")
    ] = 0,
    azimuthal_index: Annotated[
        int,
        typer.Option(help="Azimuthal mode index l of the LG beam (topological charge)"),
    ] = 1,
    polarization_handedness: Annotated[
        str,
        typer.Option(
            help="Circular polarization handedness relative to azimuthal index l: 'opposite' (spin-orbit cancellation j_z = l + sigma = 0, Hu et al. 2018 disks) or 'same' (j_z = l + sigma != 0, helical bunches and vortex radiation)"
        ),
    ] = "opposite",
    polarization_type: Annotated[
        str,
        typer.Option(
            help="Explicit laser polarization override: 'auto' (use polarization_handedness), 'right-circular', 'left-circular', 'linear-x', 'linear-y'"
        ),
    ] = "auto",
    waist_radius_lambda: Annotated[
        float,
        typer.Option(
            help="Beam waist radius w_0 in units of laser wavelength lambda_0"
        ),
    ] = 3.0,
    fiber_radius_lambda: Annotated[
        float,
        typer.Option(help="Nanofiber radius R_0 in units of laser wavelength lambda_0"),
    ] = 0.5,
    fiber_length_lambda: Annotated[
        float,
        typer.Option(help="Nanofiber length L_0 in units of laser wavelength lambda_0"),
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
    pulse_duration_periods: Annotated[
        float,
        typer.Option(
            help="Laser pulse duration parameter tau in units of laser periods T_0"
        ),
    ] = 5.0,
    t_final_periods: Annotated[
        float, typer.Option(help="Final simulation time in units of laser periods T_0")
    ] = 35.0,
    snapshot_periods: Annotated[
        str,
        typer.Option(
            help="Comma-separated laser period timestamps at which to record and plot bunch snapshots (e.g. '10.0,20.0,30.0')"
        ),
    ] = "10.0,20.0,30.0",
    steps_per_period: Annotated[
        int, typer.Option(help="Number of time integration steps per laser period T_0")
    ] = 100,
    exact_paraxial: Annotated[
        bool,
        typer.Option(
            help="Use exact paraxial analytical derivatives for longitudinal EM fields (Ez, Bz)"
        ),
    ] = True,
    compute_radiation: Annotated[
        bool,
        typer.Option(
            help="Accumulate coherent scattered radiation on detector throughout simulation"
        ),
    ] = True,
    detector_distance_lambda: Annotated[
        float,
        typer.Option(
            help="Detector distance z_det from entrance face in laser wavelengths lambda_0"
        ),
    ] = 100.0,
    detector_width_waists: Annotated[
        float,
        typer.Option(
            help="Detector transverse width and height in laser waist radii w_0"
        ),
    ] = 15.0,
    detector_grid_size: Annotated[
        int, typer.Option(help="Detector pixel grid resolution (grid_size x grid_size)")
    ] = 32,
    radiation_harmonic: Annotated[
        int,
        typer.Option(
            help="Harmonic order m of laser frequency for detector radiation (omega = m * omega_0)"
        ),
    ] = 1,
    radiation_frequency: Annotated[
        float,
        typer.Option(
            help="Explicit radiation frequency in a.u. (if 0.0, defaults to radiation_harmonic * omega_0)"
        ),
    ] = 0.0,
    spectrum_num_frequencies: Annotated[
        int,
        typer.Option(help="Number of frequency sample points for radiation spectrum"),
    ] = 64,
    spectrum_window_fraction: Annotated[
        float,
        typer.Option(
            help="Fractional bandwidth window around radiation frequency for spectrum"
        ),
    ] = 1.0,
    spectrum_x_offset_waists: Annotated[
        float,
        typer.Option(
            help="Transverse x-offset for spectrum observer point in laser waist radii w_0"
        ),
    ] = 0.5,
    seed: Annotated[int, typer.Option(help="Random number generator seed")] = 42,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Overwrite last run directory instead of creating a timestamped run directory"
        ),
    ] = False,
) -> None:
    """Simulates attosecond electron bunch generation from a nanofiber target
    driven by an intense circularly polarized Laguerre-Gaussian (LG) laser pulse,
    replicating the scheme of Hu et al., Scientific Reports 8:7282 (2018),
    and evaluates coherent scattered radiation and orbital angular momentum (OAM) transfer.
    """
    run_datetime = datetime.now(UTC).astimezone()
    print(
        f"[bold green]Started nanofiber attosecond bunch simulation at {run_datetime}[/bold green]"
    )

    initialize_jax()

    laser_frequency = DEFAULT_LASER_FREQUENCY
    laser_wavelength = (2 * pi * c) / laser_frequency
    laser_period = (2 * pi) / laser_frequency

    amplitude = a_0 * m_e * c * laser_frequency / abs(q)
    waist_radius = waist_radius_lambda * laser_wavelength
    fiber_radius = fiber_radius_lambda * laser_wavelength
    fiber_length = fiber_length_lambda * laser_wavelength

    if polarization_type != "auto":
        polarization, sigma = get_polarization_by_name(polarization_type)
    else:
        polarization, sigma = resolve_circular_polarization(
            azimuthal_index, polarization_handedness
        )

    j_z = azimuthal_index + sigma
    print(
        f"Beam mode: azimuthal index l = {azimuthal_index}, spin sigma = {sigma} => total angular momentum j_z = {j_z}"
    )
    if j_z == 0:
        print(
            "[cyan]j_z = 0: Spin-orbit cancellation gives cylindrically symmetric transverse force (Hu et al. 2018). Bunches form axisymmetric disks with L_z^EM = 0.[/cyan]"
        )
    else:
        print(
            f"[magenta]j_z = {j_z} != 0: Helical transverse force creates |j_z| = {abs(j_z)} intertwined spiral bunches. Radiation carries orbital angular momentum.[/magenta]"
        )

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
    print("Laser parameters:", laser_parameters)

    tau_0 = pulse_duration_periods * laser_period
    pulse_parameters = CosineSquaredPulseParameters(phi_0=-tau_0, tau_0=tau_0)
    print("Pulse parameters:", pulse_parameters)

    generator = np.random.default_rng(seed)
    skin_depth = skin_depth_lambda * laser_wavelength
    if skin_depth_lambda > 0.0:
        print(
            f"Generating {num_electrons} initial electrons within fiber skin depth {skin_depth_lambda:.3f} lambda_0"
        )
    else:
        print(
            f"Generating {num_electrons} initial electrons uniformly within nanofiber volume (R_0={fiber_radius_lambda} lambda_0, L_0={fiber_length_lambda} lambda_0)"
        )

    initial_positions_3d = generate_nanofiber_initial_positions(
        generator,
        fiber_radius,
        fiber_length,
        num_electrons,
        skin_depth=skin_depth,
        z_start=0.0,
    )
    initial_positions = to_position_four_vectors(initial_positions_3d)
    initial_momenta = generate_initial_particle_momenta_at_rest(num_electrons, m_e)

    dt = laser_period / steps_per_period
    print(
        f"Time step dt = {dt:.4f} a.u. ({steps_per_period} steps per laser period T_0 = {laser_period:.2f} a.u.)"
    )

    parsed_snapshots, schedule_times = parse_snapshot_schedule(
        snapshot_periods, laser_period, t_final_periods
    )

    output_directory = prepare_output_directory(
        "nanofiber_attosecond_bunches",
        run_datetime,
        overwrite=overwrite,
        overwrite_run_tag=f"jz_{j_z}",
        timestamp_suffix=f"jz{j_z}",
    )

    detector_z = detector_distance_lambda * laser_wavelength
    det_params = DetectorParameters(
        width=detector_width_waists * waist_radius,
        height=detector_width_waists * waist_radius,
        grid_size_x=detector_grid_size,
        grid_size_y=detector_grid_size,
    )
    det_positions_flat = initialize_detector_positions_negative_z(
        det_params, z_distance=detector_z
    )
    det_positions_jnp = jnp.asarray(det_positions_flat)

    rad_freq = (
        radiation_frequency
        if radiation_frequency > 0.0
        else radiation_harmonic * laser_frequency
    )

    spectrum_pos = jnp.array([spectrum_x_offset_waists * waist_radius, 0.0, detector_z])
    freq_min = max(
        0.01 * laser_frequency, rad_freq * (1.0 - spectrum_window_fraction / 2)
    )
    freq_max = rad_freq * (1.0 + spectrum_window_fraction / 2)
    frequencies_np = np.linspace(freq_min, freq_max, spectrum_num_frequencies)
    frequencies_jnp = jnp.asarray(frequencies_np)

    num_det_pixels = detector_grid_size * detector_grid_size
    current_detector_E = jnp.zeros((num_det_pixels, 3), dtype=jnp.complex128)
    current_detector_B = jnp.zeros((num_det_pixels, 3), dtype=jnp.complex128)
    current_spectrum_E = jnp.zeros((spectrum_num_frequencies, 3), dtype=jnp.complex128)

    configuration = {
        "a_0": a_0,
        "laser_parameters": laser_parameters,
        "pulse_parameters": pulse_parameters,
        "polarization_handedness": polarization_handedness,
        "polarization_type": polarization_type,
        "j_z": j_z,
        "fiber_radius_lambda": fiber_radius_lambda,
        "fiber_length_lambda": fiber_length_lambda,
        "skin_depth_lambda": skin_depth_lambda,
        "num_electrons": num_electrons,
        "laser_period": laser_period,
        "time_step": dt,
        "t_final_periods": t_final_periods,
        "snapshots_periods": parsed_snapshots,
        "detector_distance_lambda": detector_distance_lambda,
        "detector_width_waists": detector_width_waists,
        "detector_grid_size": detector_grid_size,
        "radiation_harmonic": radiation_harmonic,
        "radiation_frequency": rad_freq,
    }
    save_json(output_directory / "configuration.json", configuration)

    print(
        f"[bold blue]Starting simulation from t = 0 to t = {t_final_periods} T_0...[/bold blue]"
    )
    sim_start_time = perf_counter()

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
        ) = integrate_interval(
            current_positions,
            current_momenta,
            initial_positions_jnp,
            det_positions_jnp,
            current_detector_E,
            current_detector_B,
            rad_freq,
            spectrum_pos,
            frequencies_jnp,
            current_spectrum_E,
            num_steps,
            dt,
            laser_parameters,
            pulse_parameters,
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

    sim_end_time = perf_counter()
    print(
        f"[bold green]Simulation completed in {sim_end_time - sim_start_time:.2f} seconds![/bold green]"
    )

    print("Generating bunch snapshot plots...")
    save_nanofiber_snapshot_plots(
        output_directory,
        recorded_snapshots,
        laser_wavelength,
        fiber_radius,
        fiber_length,
        waist_radius,
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
            "[bold blue]Analyzing scattered radiation and orbital angular momentum on detector...[/bold blue]"
        )
        current_detector_E.block_until_ready()
        det_pos_2d = det_params.reshape_to_grid(np.asarray(det_positions_flat))
        det_E_2d = det_params.reshape_to_grid(np.asarray(current_detector_E))
        det_B_2d = det_params.reshape_to_grid(np.asarray(current_detector_B))

        dL_z = compute_angular_momentum_derivative(det_pos_2d, det_E_2d, rad_freq)
        total_radiated_Lz = integrate_detector_oam_spectral_density(det_pos_2d, dL_z)
        print(
            f"Total electromagnetic radiated orbital angular momentum L_z^EM: {total_radiated_Lz:.6e} a.u."
        )

        fig = plt.figure(dpi=200)
        plot_detector_radiation_and_oam(
            fig,
            det_pos_2d,
            det_E_2d,
            dL_z,
            laser_wavelength,
            title=f"Detector radiation at $\\omega = {rad_freq / laser_frequency:.1f} \\omega_0$ ($l={azimuthal_index}, \\sigma={sigma} \\to j_z={j_z}$)",
        )
        save_figure(
            fig, output_directory / "detector_radiation_and_oam", formats=("png", "pdf")
        )

        spec_E_np = np.asarray(current_spectrum_E)
        fig = plt.figure(dpi=200)
        plot_scattered_radiation_spectrum(fig, rad_freq, frequencies_np, spec_E_np)
        save_figure(
            fig,
            output_directory / "scattered_radiation_spectrum",
            formats=("pdf", "png"),
        )

        np.save(output_directory / "detector_positions.npy", det_pos_2d)
        np.save(output_directory / "detector_electric_field.npy", det_E_2d)
        np.save(output_directory / "detector_magnetic_field.npy", det_B_2d)
        np.save(output_directory / "detector_dL_z.npy", dL_z)
        np.save(output_directory / "spectrum_frequencies.npy", frequencies_np)
        np.save(output_directory / "spectrum_electric_field.npy", spec_E_np)

        radiation_summary = {
            "radiation_frequency": rad_freq,
            "radiation_harmonic": rad_freq / laser_frequency,
            "total_radiated_Lz": total_radiated_Lz,
            "max_abs_dL_z": float(np.max(np.abs(dL_z))),
            "detector_distance_lambda": detector_distance_lambda,
            "detector_width_waists": detector_width_waists,
            "detector_grid_size": detector_grid_size,
            "j_z": j_z,
            "azimuthal_index": azimuthal_index,
            "polarization_sigma": sigma,
        }
        save_json(output_directory / "radiation_summary.json", radiation_summary)

    print(
        f"[bold green]All results and figures saved to: '{output_directory}'[/bold green]"
    )


if __name__ == "__main__":
    typer.run(main)
