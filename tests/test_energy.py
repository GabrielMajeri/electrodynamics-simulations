import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from scipy.special import eval_genlaguerre

from electrodynamics.beams import (
    GaussianBeamParameters,
    LaguerreGaussBeamParameters,
    LaserParameters,
)
from electrodynamics.constants import (
    ATOMIC_TIME_S,
    BOHR_RADIUS_M,
    DEFAULT_LASER_FREQUENCY,
    ELEMENTARY_CHARGE_C,
    HARTREE_TO_EV,
    HARTREE_TO_JOULES,
    au_energy_to_ev,
    au_energy_to_joules,
    au_length_to_meters,
    au_time_to_seconds,
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
    DiskDetectorParameters,
    build_detector_grid,
)
from electrodynamics.energy import (
    accumulate_larmor_radiated_energy,
    analyze_energy_transfer,
    build_energy_summary_dict,
    build_frequency_quadrature,
    compute_detector_spectral_energy,
    compute_detector_total_energy,
    compute_effective_bandwidth,
    compute_electron_kinetic_energy,
    compute_electron_kinetic_energy_balance,
    compute_factorized_detector_energy,
    compute_incident_beam_energy,
    compute_incident_beam_energy_numerical,
    compute_incident_beam_fluence,
    compute_incident_beam_interception_fraction,
    compute_larmor_radiated_energy_step,
    compute_pulse_energy_integral,
    compute_spectral_energy_flux,
    compute_thin_sheet_reflectivity,
    format_energy_transfer_report,
)
from electrodynamics.fields import (
    compute_coherent_detector_fields_multifreq,
    compute_relativistic_reflection_parameters,
    compute_scattered_electric_and_magnetic_fields,
)
from electrodynamics.initial_conditions import (
    generate_initial_particle_momenta_at_rest,
    generate_initial_particle_momenta_moving_towards_laser,
    generate_initial_positions_on_thin_sheet,
    to_position_four_vectors,
)
from electrodynamics.integrate import (
    integration_step_rk4,
    integration_step_rk4_lab_time,
)
from electrodynamics.polarization import Polarizations
from electrodynamics.pulse import (
    CosineSquaredPulseParameters,
    GaussianPulseParameters,
    PulseWithFlatPeakParameters,
)
from electrodynamics.serialization import save_json


def _eval_orthonormal_lg_mode(
    x: np.ndarray, y: np.ndarray, waist: float, p: int, l: int
) -> np.ndarray:
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    abs_l = abs(l)
    norm = (
        math.sqrt(2.0 * math.factorial(p) / (math.pi * math.factorial(p + abs_l)))
        / waist
    )
    xi = 2.0 * (r / waist) ** 2
    return (
        norm
        * ((math.sqrt(2.0) * r / waist) ** abs_l)
        * eval_genlaguerre(p, abs_l, xi)
        * np.exp(-(r**2) / (waist**2))
        * np.exp(1j * l * phi)
    )


def test_si_constants_and_unit_conversions() -> None:
    assert BOHR_RADIUS_M == pytest.approx(5.29177210544e-11, rel=1e-12)
    assert ATOMIC_TIME_S == pytest.approx(2.4188843265864e-17, rel=1e-12)
    assert HARTREE_TO_JOULES == pytest.approx(4.3597447222060e-18, rel=1e-12)
    assert ELEMENTARY_CHARGE_C == pytest.approx(1.602176634e-19, rel=1e-12)
    assert HARTREE_TO_JOULES / ELEMENTARY_CHARGE_C == pytest.approx(
        HARTREE_TO_EV, rel=1e-7
    )

    assert au_length_to_meters(2.5) == pytest.approx(2.5 * BOHR_RADIUS_M)
    assert au_time_to_seconds(4.0) == pytest.approx(4.0 * ATOMIC_TIME_S)
    assert au_energy_to_joules(3.0) == pytest.approx(3.0 * HARTREE_TO_JOULES)
    assert au_energy_to_ev(3.0) == pytest.approx(3.0 * HARTREE_TO_EV)


def test_multifreq_kernel_matches_single_freq_reference(jax_config) -> None:
    rng = np.random.default_rng(7)
    num_particles = 12
    origin_pos = jnp.asarray(
        np.column_stack(
            [
                np.zeros(num_particles),
                rng.normal(0.0, 5.0, size=(num_particles, 3)),
            ]
        )
    )
    pos = origin_pos + jnp.asarray(
        np.column_stack(
            [
                np.full(num_particles, 120.0),
                rng.normal(0.0, 1.5, size=(num_particles, 3)),
            ]
        )
    )

    gamma = 3.0
    beta_vec = rng.normal(0.0, 0.25, size=(num_particles, 3))
    beta_vec[:, 2] = -np.sqrt(
        max(0.0, 1.0 - 1.0 / gamma**2) - np.sum(beta_vec[:, :2] ** 2, axis=-1)
    )
    gamma_col = 1.0 / np.sqrt(1.0 - np.sum(beta_vec**2, axis=-1, keepdims=True))
    mom = jnp.asarray(
        np.concatenate([gamma_col * c, gamma_col * c * beta_vec], axis=-1)
    )
    init_mom = jnp.asarray(
        np.repeat(
            [[gamma * c, 0.0, 0.0, -np.sqrt(gamma**2 - 1.0) * c]], num_particles, axis=0
        )
    )
    freqs = jnp.array([0.04, 0.057, 0.171, 0.24])

    def reference_fields(det_positions, is_lab_time, initial_momentum, second_order):
        def one_freq(w):
            def one_pix(dp):
                E, B = compute_scattered_electric_and_magnetic_fields(
                    w,
                    pos,
                    mom,
                    origin_pos,
                    dp,
                    include_second_order_terms=second_order,
                    is_lab_time=is_lab_time,
                    initial_momentum=initial_momentum,
                )
                return jnp.sum(E, axis=0), jnp.sum(B, axis=0)

            return jax.vmap(one_pix)(det_positions)

        E_fw, B_fw = jax.vmap(one_freq)(freqs)
        return jnp.swapaxes(E_fw, 0, 1), jnp.swapaxes(B_fw, 0, 1)

    for num_pix, batch_sz in ((5, 64), (70, 32), (17, None)):
        det_pos = jnp.asarray(
            np.column_stack(
                [
                    rng.uniform(-80.0, 80.0, size=(num_pix, 2)),
                    np.full(num_pix, -5000.0),
                ]
            )
        )
        for is_lab in (True, False):
            for init_m in (None, init_mom):
                for second_order in (False, True):
                    E_fused, B_fused = compute_coherent_detector_fields_multifreq(
                        pos,
                        mom,
                        origin_pos,
                        det_pos,
                        freqs,
                        is_lab_time=is_lab,
                        initial_momentum=init_m,
                        include_second_order_terms=second_order,
                        pixel_batch_size=batch_sz,
                    )
                    E_ref, B_ref = reference_fields(
                        det_pos, is_lab, init_m, second_order
                    )
                    assert E_fused.shape == (num_pix, len(freqs), 3)
                    assert B_fused.shape == (num_pix, len(freqs), 3)
                    np.testing.assert_allclose(E_fused, E_ref, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(B_fused, B_ref, rtol=1e-12, atol=1e-12)

    # Constant-velocity drift subtraction cancels to zero
    det_pos_const = jnp.array([[10.0, -20.0, -5000.0], [0.0, 5.0, -5000.0]])
    for is_lab in (True, False):
        E_zero, B_zero = compute_coherent_detector_fields_multifreq(
            pos,
            init_mom,
            origin_pos,
            det_pos_const,
            freqs,
            is_lab_time=is_lab,
            initial_momentum=init_mom,
        )
        np.testing.assert_allclose(E_zero, 0.0, atol=1e-15)
        np.testing.assert_allclose(B_zero, 0.0, atol=1e-15)


def test_spectral_energy_total_and_factorized_energy() -> None:
    quad = build_frequency_quadrature(
        omega_center=0.2, bandwidth_fraction=0.6, num_frequencies=16
    )
    assert quad.omega_min == pytest.approx(0.14)
    assert quad.omega_max == pytest.approx(0.26)
    assert np.sum(quad.weights) == pytest.approx(0.12, rel=1e-13)

    quad_explicit = build_frequency_quadrature(
        0.0, num_frequencies=12, omega_min=0.1, omega_max=0.3
    )
    assert quad_explicit.omega_center == pytest.approx(0.2)
    assert quad_explicit.bandwidth_fraction == pytest.approx(1.0)
    assert np.sum(quad_explicit.weights) == pytest.approx(0.2, rel=1e-13)

    with pytest.raises(ValueError):
        build_frequency_quadrature(0.2, bandwidth_fraction=2.0)
    with pytest.raises(ValueError):
        build_frequency_quadrature(0.2, omega_min=0.1)

    quad_fine = build_frequency_quadrature(
        omega_center=0.2, bandwidth_fraction=0.6, num_frequencies=33
    )

    disk = DiskDetectorParameters(
        radius=30.0, num_radial=40, num_azimuthal=48, radial_sampling="gauss-legendre"
    )
    grid = build_detector_grid(disk, distance=1e6, direction=-1)
    x = grid.local_coords_2d[..., 0]
    y = grid.local_coords_2d[..., 1]
    u_mode = _eval_orthonormal_lg_mode(x, y, waist=4.0, p=1, l=-1)

    # Exact Gaussian spectral profile S(omega) = exp(-(omega - omega_c)^2 / sigma^2)
    sigma_w = 0.015
    S_omega = np.exp(-((quad_fine.frequencies - 0.2) ** 2) / (sigma_w**2))
    # Factorized far-field wave propagating along -z_hat: B = (-z_hat) x E
    E_2d_single = np.zeros((*grid.grid_shape, 3), dtype=np.complex128)
    E_2d_single[..., 0] = u_mode
    B_2d_single = np.zeros_like(E_2d_single)
    B_2d_single[..., 1] = -u_mode

    flux_single = compute_spectral_energy_flux(
        E_2d_single, B_2d_single, normal_vector=(0.0, 0.0, -1.0)
    )
    np.testing.assert_allclose(flux_single, c * np.abs(u_mode) ** 2, rtol=1e-13)

    spec_res = compute_detector_spectral_energy(
        E_2d_single,
        B_2d_single,
        grid.area_weights_2d,
        normal_vector=(0.0, 0.0, -1.0),
        detector_positions=grid.local_coords_2d,
    )
    assert spec_res.dU_domega == pytest.approx(c, rel=1e-10)
    assert spec_res.relative_far_field_mismatch < 1e-8

    E_multi = E_2d_single[:, :, None, :] * S_omega[None, None, :, None]
    B_multi = B_2d_single[:, :, None, :] * S_omega[None, None, :, None]

    total_res = compute_detector_total_energy(
        E_multi,
        B_multi,
        grid.area_weights_2d,
        quad_fine.weights,
        normal_vector=(0.0, 0.0, -1.0),
        detector_positions=grid.local_coords_2d,
    )
    expected_bw = sigma_w * math.sqrt(math.pi / 2.0)
    expected_total = c * expected_bw
    assert total_res.total_energy_au == pytest.approx(expected_total, rel=1e-8)
    assert total_res.total_energy_joules == pytest.approx(
        au_energy_to_joules(total_res.total_energy_au)
    )
    assert total_res.total_energy_ev == pytest.approx(
        au_energy_to_ev(total_res.total_energy_au)
    )
    assert total_res.relative_far_field_mismatch < 1e-8
    assert np.sum(total_res.fluence_per_pixel * grid.area_weights_2d) == pytest.approx(
        total_res.total_energy_au, rel=1e-12
    )
    assert np.sum(total_res.spectrum_dU_domega * quad_fine.weights) == pytest.approx(
        total_res.total_energy_au, rel=1e-12
    )

    spec_E = np.zeros((quad_fine.frequencies.size, 3), dtype=np.complex128)
    spec_E[:, 0] = S_omega
    bw_eff = compute_effective_bandwidth(
        quad_fine.frequencies,
        spec_E,
        reference_frequency=0.2,
        frequency_weights=quad_fine.weights,
    )
    assert bw_eff == pytest.approx(expected_bw, rel=1e-8)

    dense_freqs = np.linspace(0.14, 0.26, 257)
    dense_spec_E = np.zeros((dense_freqs.size, 3), dtype=np.complex128)
    dense_spec_E[:, 0] = np.exp(-((dense_freqs - 0.2) ** 2) / (sigma_w**2))
    bw_trap = compute_effective_bandwidth(
        dense_freqs, dense_spec_E, reference_frequency=0.2
    )
    assert bw_trap == pytest.approx(expected_bw, rel=1e-6)

    fact_res = compute_factorized_detector_energy(
        E_2d_single,
        B_2d_single,
        grid.area_weights_2d,
        quad_fine.frequencies,
        spec_E,
        reference_frequency=0.2,
        normal_vector=(0.0, 0.0, -1.0),
        spectrum_frequency_weights=quad_fine.weights,
    )
    assert fact_res.total_energy_au == pytest.approx(
        total_res.total_energy_au, rel=1e-8
    )


def test_incident_beam_energy_analytical_vs_numerical(jax_config) -> None:
    omega_0 = DEFAULT_LASER_FREQUENCY
    lambda_0 = (2.0 * math.pi * c) / omega_0
    period_0 = (2.0 * math.pi) / omega_0
    a_0 = 0.05
    E_0 = a_0 * m_e * c * omega_0 / abs(q)
    w_0 = 8.0 * lambda_0
    tau_0 = 4.0 * period_0

    pulses = (
        GaussianPulseParameters(phi_0=0.0, tau_0=tau_0),
        CosineSquaredPulseParameters(phi_0=0.0, tau_0=tau_0),
        PulseWithFlatPeakParameters(phi_0=0.0, tau_0=tau_0, peak_duration_periods=2),
    )

    gauss_lin = GaussianBeamParameters(
        omega_0, lambda_0, E_0, Polarizations.LINEAR_X.value, w_0
    )
    gauss_circ = GaussianBeamParameters(
        omega_0, lambda_0, E_0, Polarizations.RIGHT_CIRCULAR.value, w_0
    )
    lg_00 = LaguerreGaussBeamParameters(
        omega_0, lambda_0, E_0, Polarizations.LINEAR_X.value, w_0, 0, 0
    )
    lg_01 = LaguerreGaussBeamParameters(
        omega_0, lambda_0, E_0, Polarizations.RIGHT_CIRCULAR.value, w_0, 0, 1
    )
    lg_1m2 = LaguerreGaussBeamParameters(
        omega_0, lambda_0, E_0, Polarizations.LINEAR_X.value, w_0, 1, -2
    )

    for pulse in pulses:
        u_gauss = compute_incident_beam_energy(gauss_lin, pulse)
        u_lg00 = compute_incident_beam_energy(lg_00, pulse)
        # Intentional normalization difference: LG(0, 0) has exactly 2x GaussianBeamParameters energy
        assert u_lg00 == pytest.approx(2.0 * u_gauss, rel=1e-12)

        for beam in (gauss_lin, gauss_circ, lg_00, lg_01, lg_1m2):
            u_ana = compute_incident_beam_energy(beam, pulse)
            u_num = compute_incident_beam_energy_numerical(beam, pulse)
            assert u_num == pytest.approx(u_ana, rel=1e-4)

    # Plane wave over finite circular aperture (normal and oblique)
    plane = LaserParameters(omega_0, lambda_0, E_0, Polarizations.LINEAR_X.value)
    r_ap = 3.0 * lambda_0
    cos_pulse = CosineSquaredPulseParameters(phi_0=0.0, tau_0=tau_0)
    assert compute_pulse_energy_integral(cos_pulse) == pytest.approx(0.75 * tau_0)
    assert compute_incident_beam_fluence(plane, cos_pulse) == pytest.approx(
        (c / (8.0 * math.pi)) * (E_0**2) * 0.75 * tau_0
    )
    with pytest.raises(ValueError):
        compute_incident_beam_energy(plane, cos_pulse)

    for theta_in in (0.0, math.pi / 6.0):
        u_plane_ana = compute_incident_beam_energy(
            plane, cos_pulse, aperture_radius=r_ap, incidence_angle=theta_in
        )
        u_plane_num = compute_incident_beam_energy_numerical(
            plane, cos_pulse, aperture_radius=r_ap, incidence_angle=theta_in
        )
        assert u_plane_num == pytest.approx(u_plane_ana, rel=2e-4)

    # Oblique invariance for full beam & finite-aperture interception fraction
    z_test = 2.0 * lambda_0
    u_oblique_full_ana = compute_incident_beam_energy(
        lg_01, cos_pulse, incidence_angle=math.pi / 6.0
    )
    u_oblique_full_num = compute_incident_beam_energy_numerical(
        lg_01, cos_pulse, z_plane=z_test, incidence_angle=math.pi / 6.0
    )
    assert u_oblique_full_num == pytest.approx(u_oblique_full_ana, rel=2e-4)

    r_clip = 0.9 * w_0
    for theta_in in (0.0, math.pi / 6.0):
        u_clip_ana = compute_incident_beam_energy(
            lg_01,
            cos_pulse,
            aperture_radius=r_clip,
            z_plane=z_test,
            incidence_angle=theta_in,
        )
        u_clip_num = compute_incident_beam_energy_numerical(
            lg_01,
            cos_pulse,
            aperture_radius=r_clip,
            z_plane=z_test,
            incidence_angle=theta_in,
        )
        assert u_clip_num == pytest.approx(u_clip_ana, rel=5e-4)

    # Large-aperture limit of interception fraction approaches 1.0 accurately
    assert compute_incident_beam_interception_fraction(
        lg_1m2, aperture_radius=100.0 * w_0
    ) == pytest.approx(1.0, abs=1e-10)


def test_single_electron_thomson_energy_conservation(jax_config) -> None:
    # Kinetic energy balance helper checks
    mom_rest = generate_initial_particle_momenta_at_rest(2, m_e)
    assert compute_electron_kinetic_energy(mom_rest) == pytest.approx(0.0, abs=1e-12)

    gamma_rel = 5.0
    mom_rel = generate_initial_particle_momenta_moving_towards_laser(2, gamma_rel, m_e)
    expected_ke = 2.0 * (gamma_rel - 1.0) * m_e * (c**2)
    assert compute_electron_kinetic_energy(mom_rel) == pytest.approx(
        expected_ke, rel=1e-12
    )
    ke_bal = compute_electron_kinetic_energy_balance(mom_rest, mom_rel)
    assert ke_bal.delta_au == pytest.approx(expected_ke, rel=1e-12)
    assert ke_bal.mean_final_gamma == pytest.approx(gamma_rel, rel=1e-12)

    # N_e = 1 Thomson scattering in weak linearly polarized plane wave
    omega_0 = DEFAULT_LASER_FREQUENCY
    lambda_0 = (2.0 * math.pi * c) / omega_0
    period_0 = (2.0 * math.pi) / omega_0
    a_0 = 1e-3
    E_0 = a_0 * m_e * c * omega_0 / abs(q)
    tau_0 = 6.0 * period_0

    laser = LaserParameters(omega_0, lambda_0, E_0, Polarizations.LINEAR_X.value)
    pulse = CosineSquaredPulseParameters(phi_0=tau_0, tau_0=tau_0)

    steps_per_period = 60
    dt = period_0 / steps_per_period
    num_steps = round((2.0 * tau_0) / dt)

    pos0 = jnp.asarray(to_position_four_vectors(np.array([[0.0, 0.0, 0.0]])))
    mom0 = jnp.asarray(generate_initial_particle_momenta_at_rest(1, m_e))

    def scan_step(carry, _):
        pos, mom, u_larm = carry
        new_pos, new_mom = integration_step_rk4_lab_time(pos, mom, dt, laser, pulse)
        d_larm = compute_larmor_radiated_energy_step(
            new_pos, new_mom, dt, laser, pulse, is_lab_time=True
        )
        return (new_pos, new_mom, u_larm + d_larm), (new_pos[0], new_mom[0])

    (_, _, u_larmor_step), (pos_hist, mom_hist) = jax.lax.scan(
        scan_step,
        (pos0, mom0, jnp.zeros((1,), dtype=jnp.float64)),
        None,
        length=num_steps,
    )
    u_larmor_acc = accumulate_larmor_radiated_energy(
        pos_hist[:, None, :], mom_hist[:, None, :], dt, laser, pulse, is_lab_time=True
    )
    assert float(u_larmor_acc[0]) == pytest.approx(float(u_larmor_step[0]), rel=1e-4)

    sigma_thomson = (8.0 * math.pi / 3.0) * (((q**2) / (m_e * c**2)) ** 2)
    f_inc = compute_incident_beam_fluence(laser, pulse)
    u_thomson_theory = sigma_thomson * f_inc
    assert float(u_larmor_step[0]) == pytest.approx(u_thomson_theory, rel=5e-3)

    # Evaluate multi-frequency scattered field from the recorded trajectory
    quad = build_frequency_quadrature(
        omega_center=omega_0, bandwidth_fraction=1.2, num_frequencies=40
    )
    origin_rep = jnp.repeat(pos0, num_steps, axis=0)
    freqs_jnp = jnp.asarray(quad.frequencies)

    # 1) Backward disk detector: R_disk = 2 * R_det => theta_max = atan(2)
    r_det = 2000.0 * lambda_0
    disk = DiskDetectorParameters(
        radius=2.0 * r_det,
        num_radial=32,
        num_azimuthal=48,
        radial_sampling="gauss-legendre",
    )
    grid = build_detector_grid(disk, distance=r_det, direction=-1)
    E_disk, B_disk = compute_coherent_detector_fields_multifreq(
        pos_hist,
        mom_hist,
        origin_rep,
        jnp.asarray(grid.lab_positions_flat),
        freqs_jnp,
        is_lab_time=True,
        initial_momentum=jnp.repeat(mom0, num_steps, axis=0),
    )
    E_disk_np = np.asarray(E_disk) * dt
    B_disk_np = np.asarray(B_disk) * dt

    disk_energy = compute_detector_total_energy(
        E_disk_np,
        B_disk_np,
        grid.area_weights_2d.reshape(-1),
        quad.weights,
        normal_vector=grid.normal_lab,
        detector_positions=grid.lab_positions_flat,
    )
    theta_max = math.atan(2.0)
    cos_t = math.cos(theta_max)
    dipole_frac = 0.5 - (3.0 / 8.0) * cos_t - (1.0 / 8.0) * (cos_t**3)
    assert disk_energy.relative_far_field_mismatch < 1e-3
    assert disk_energy.total_energy_au == pytest.approx(
        dipole_frac * float(u_larmor_step[0]), rel=1e-2
    )

    # 2) Full 4*pi Gauss-Legendre spherical shell
    n_mu, n_phi = 24, 36
    mu, w_mu = np.polynomial.legendre.leggauss(n_mu)
    sin_theta = np.sqrt(1.0 - mu**2)
    phi = (2.0 * math.pi * np.arange(n_phi, dtype=np.float64)) / n_phi
    dphi = (2.0 * math.pi) / n_phi

    normals_sphere = np.stack(
        [
            sin_theta[:, None] * np.cos(phi)[None, :],
            sin_theta[:, None] * np.sin(phi)[None, :],
            np.broadcast_to(mu[:, None], (n_mu, n_phi)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    pos_sphere = r_det * normals_sphere
    area_sphere = np.broadcast_to(
        (r_det**2) * w_mu[:, None] * dphi, (n_mu, n_phi)
    ).reshape(-1)

    E_sph, B_sph = compute_coherent_detector_fields_multifreq(
        pos_hist,
        mom_hist,
        origin_rep,
        jnp.asarray(pos_sphere),
        freqs_jnp,
        is_lab_time=True,
        initial_momentum=jnp.repeat(mom0, num_steps, axis=0),
    )
    sphere_energy = compute_detector_total_energy(
        np.asarray(E_sph) * dt,
        np.asarray(B_sph) * dt,
        area_sphere,
        quad.weights,
        normal_vector=normals_sphere,
        detector_positions=pos_sphere,
    )
    assert sphere_energy.total_energy_au == pytest.approx(
        float(u_larmor_step[0]), rel=1e-2
    )

    # 3) Relativistic (gamma = 2) Larmor benchmark in both lab-time and proper-time
    gamma_e = 2.0
    beta_e = math.sqrt(1.0 - 1.0 / gamma_e**2)
    t_coll = (tau_0 + 0.5 * period_0) / (1.0 + beta_e)
    pulse_rel = CosineSquaredPulseParameters(phi_0=t_coll, tau_0=tau_0)
    pos_rel0 = jnp.asarray(
        to_position_four_vectors(np.array([[0.0, 0.0, beta_e * c * t_coll]]))
    )
    mom_rel0 = jnp.asarray(
        generate_initial_particle_momenta_moving_towards_laser(1, gamma_e, m_e)
    )

    dt_rel = period_0 / (60.0 * (1.0 + beta_e))
    num_steps_rel = round((2.0 * t_coll) / dt_rel)
    _, _, u_larmor_rel_lab = jax.lax.scan(
        lambda carry, _: (
            (
                *integration_step_rk4_lab_time(
                    carry[0], carry[1], dt_rel, laser, pulse_rel
                ),
                carry[2]
                + compute_larmor_radiated_energy_step(
                    carry[0], carry[1], dt_rel, laser, pulse_rel, is_lab_time=True
                ),
            ),
            None,
        ),
        (pos_rel0, mom_rel0, jnp.zeros((1,), dtype=jnp.float64)),
        None,
        length=num_steps_rel,
    )[0]

    dtau = dt_rel / gamma_e
    _, _, u_larmor_rel_prop = jax.lax.scan(
        lambda carry, _: (
            (
                *integration_step_rk4(carry[0], carry[1], dtau, laser, pulse_rel),
                carry[2]
                + compute_larmor_radiated_energy_step(
                    carry[0], carry[1], dtau, laser, pulse_rel, is_lab_time=False
                ),
            ),
            None,
        ),
        (pos_rel0, mom_rel0, jnp.zeros((1,), dtype=jnp.float64)),
        None,
        length=num_steps_rel,
    )[0]

    u_rel_theory = (gamma_e**2) * (1.0 + beta_e) * sigma_thomson * f_inc
    assert float(u_larmor_rel_lab[0]) == pytest.approx(u_rel_theory, rel=1e-2)
    assert float(u_larmor_rel_prop[0]) == pytest.approx(u_rel_theory, rel=1e-2)


def test_flying_mirror_multi_electron_energy_and_analysis(
    jax_config, tmp_path: Path
) -> None:
    omega_0 = DEFAULT_LASER_FREQUENCY
    lambda_0 = (2.0 * math.pi * c) / omega_0
    period_0 = (2.0 * math.pi) / omega_0
    a_0 = 0.01
    E_0 = a_0 * m_e * c * omega_0 / abs(q)
    w_0 = 5.0 * lambda_0

    laser = LaguerreGaussBeamParameters(
        omega_0,
        lambda_0,
        E_0,
        Polarizations.LINEAR_X.value,
        w_0,
        radial_index=0,
        azimuthal_index=1,
    )

    beta = 0.5
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    _, _, doppler_factor, _ = compute_relativistic_reflection_parameters(gamma, 0.0)
    assert doppler_factor == pytest.approx(3.0, rel=1e-12)
    omega_1 = doppler_factor * omega_0
    lambda_1 = lambda_0 / doppler_factor

    tau_0 = 6.0 * period_0
    t_coll = (tau_0 + 1.5 * period_0) / (1.0 + beta)
    pulse = CosineSquaredPulseParameters(phi_0=t_coll, tau_0=tau_0)

    sheet_radius = 10.0 * lambda_0
    num_electrons = 1024  # 32 x 32 polar grid
    init_pos = jnp.asarray(
        to_position_four_vectors(
            generate_initial_positions_on_thin_sheet(
                disk_radius=sheet_radius,
                num_particles=num_electrons,
                z_center=beta * c * t_coll,
                sampling_method="polar-grid",
            )
        )
    )
    init_mom = jnp.asarray(
        generate_initial_particle_momenta_moving_towards_laser(
            num_electrons, gamma, m_e
        )
    )

    z_r1 = (math.pi * w_0**2) / lambda_1
    r_det = 20.0 * z_r1
    w_1_det = w_0 * math.sqrt(1.0 + (r_det / z_r1) ** 2)
    disk = DiskDetectorParameters(
        radius=3.5 * w_1_det,
        num_radial=12,
        num_azimuthal=16,
        radial_sampling="gauss-legendre",
    )
    grid = build_detector_grid(disk, distance=r_det, direction=-1)
    det_pos_jnp = jnp.asarray(grid.lab_positions_flat)

    quad = build_frequency_quadrature(
        omega_center=omega_1, bandwidth_fraction=0.8, num_frequencies=16
    )
    freqs_jnp = jnp.asarray(quad.frequencies)
    ref_freq_jnp = jnp.array([omega_1])

    ring_offset = w_1_det / math.sqrt(2.0)
    spec_pos_jnp = jnp.array([ring_offset, 0.0, -r_det])
    spec_freqs_jnp = jnp.linspace(0.6 * omega_1, 1.4 * omega_1, 65)

    dt = period_0 / 40.0
    num_steps = round((2.0 * t_coll) / dt)

    @jdc.jit
    def run_sim():
        def step_fn(carry, _):
            pos, mom, E_mf, B_mf, E_ref, B_ref, E_spec, u_larm = carry
            new_pos, new_mom = integration_step_rk4_lab_time(pos, mom, dt, laser, pulse)

            dE_mf, dB_mf = compute_coherent_detector_fields_multifreq(
                new_pos,
                new_mom,
                init_pos,
                det_pos_jnp,
                freqs_jnp,
                is_lab_time=True,
                initial_momentum=init_mom,
                pixel_batch_size=64,
            )
            dE_ref, dB_ref = compute_coherent_detector_fields_multifreq(
                new_pos,
                new_mom,
                init_pos,
                det_pos_jnp,
                ref_freq_jnp,
                is_lab_time=True,
                initial_momentum=init_mom,
                pixel_batch_size=64,
            )

            def spec_f(w):
                E, _ = compute_scattered_electric_and_magnetic_fields(
                    w,
                    new_pos,
                    new_mom,
                    init_pos,
                    spec_pos_jnp,
                    is_lab_time=True,
                    initial_momentum=init_mom,
                )
                return jnp.sum(E, axis=0)

            dE_spec = jax.vmap(spec_f)(spec_freqs_jnp)
            d_larm = jnp.sum(
                compute_larmor_radiated_energy_step(
                    new_pos, new_mom, dt, laser, pulse, is_lab_time=True
                )
            )
            return (
                new_pos,
                new_mom,
                E_mf + dt * dE_mf,
                B_mf + dt * dB_mf,
                E_ref + dt * dE_ref[:, 0, :],
                B_ref + dt * dB_ref[:, 0, :],
                E_spec + dt * dE_spec,
                u_larm + d_larm,
            ), None

        init_carry = (
            init_pos,
            init_mom,
            jnp.zeros(
                (len(det_pos_jnp), len(quad.frequencies), 3), dtype=jnp.complex128
            ),
            jnp.zeros(
                (len(det_pos_jnp), len(quad.frequencies), 3), dtype=jnp.complex128
            ),
            jnp.zeros((len(det_pos_jnp), 3), dtype=jnp.complex128),
            jnp.zeros((len(det_pos_jnp), 3), dtype=jnp.complex128),
            jnp.zeros((len(spec_freqs_jnp), 3), dtype=jnp.complex128),
            jnp.array(0.0, dtype=jnp.float64),
        )
        return jax.lax.scan(step_fn, init_carry, None, length=num_steps)[0]

    _, final_mom, E_mf, B_mf, E_ref, B_ref, E_spec, u_larmor = run_sim()

    E_mf_2d = grid.reshape_to_grid(np.asarray(E_mf))
    B_mf_2d = grid.reshape_to_grid(np.asarray(B_mf))
    total_energy = compute_detector_total_energy(
        E_mf_2d,
        B_mf_2d,
        grid.area_weights_2d,
        quad.weights,
        normal_vector=grid.normal_lab,
        detector_positions=grid.reshape_to_grid(grid.lab_positions_flat),
    )

    fact_energy = compute_factorized_detector_energy(
        grid.reshape_to_grid(np.asarray(E_ref)),
        grid.reshape_to_grid(np.asarray(B_ref)),
        grid.area_weights_2d,
        np.asarray(spec_freqs_jnp),
        np.asarray(E_spec),
        reference_frequency=omega_1,
        normal_vector=grid.normal_lab,
    )

    u_inc = compute_incident_beam_energy(laser, pulse)
    u_inc_intercepted = compute_incident_beam_energy(
        laser, pulse, aperture_radius=sheet_radius
    )
    ke_balance = compute_electron_kinetic_energy_balance(init_mom, final_mom)

    theory = compute_thin_sheet_reflectivity(
        num_electrons=num_electrons,
        sheet_radius=sheet_radius,
        gamma=gamma,
        laser_frequency=omega_0,
        incident_energy_au=u_inc,
        doppler_factor=doppler_factor,
        simulated_detector_energy_au=total_energy.total_energy_au,
    )
    assert total_energy.relative_far_field_mismatch < 1e-4
    assert total_energy.total_energy_au == pytest.approx(
        theory.weak_theory_energy_au, rel=2e-2
    )
    assert fact_energy.total_energy_au == pytest.approx(
        total_energy.total_energy_au, rel=2e-2
    )

    analysis_by_count = analyze_energy_transfer(
        detector_energy_au=total_energy.total_energy_au,
        incident_energy_au=u_inc,
        doppler_factor=doppler_factor,
        num_electrons=num_electrons,
        detector_energy_factorized_au=fact_energy.total_energy_au,
        intercepted_incident_energy_au=u_inc_intercepted,
        larmor_energy_au=float(u_larmor),
        electron_kinetic_energy=ke_balance,
        sheet_radius=sheet_radius,
        gamma=gamma,
        laser_frequency=omega_0,
        physical_num_electrons=1.0e8,
    )
    assert analysis_by_count.laser_energy_share_per_photon == pytest.approx(1.0 / 3.0)
    assert analysis_by_count.electron_energy_share_per_photon == pytest.approx(
        2.0 / 3.0
    )
    assert analysis_by_count.coherence_gain is not None
    assert analysis_by_count.coherence_gain > 0.0
    assert analysis_by_count.saturation_electrons_reflectivity is not None
    assert analysis_by_count.saturation_electrons_kinetic is not None
    assert analysis_by_count.physical_extrapolation is not None
    assert (
        analysis_by_count.physical_extrapolation.larmor_incoherent_energy_au is not None
    )
    assert (
        analysis_by_count.physical_extrapolation.detector_energy_au
        / analysis_by_count.physical_extrapolation.larmor_incoherent_energy_au
        > 1e4
    )
    assert analysis_by_count.physical_extrapolation.test_particle_regime_valid is True

    analysis_by_charge = analyze_energy_transfer(
        detector_energy_au=total_energy.total_energy_au,
        incident_energy_au=u_inc,
        doppler_factor=doppler_factor,
        num_electrons=num_electrons,
        electron_kinetic_energy=ke_balance,
        physical_bunch_charge_pc=100.0,
    )
    assert analysis_by_charge.physical_extrapolation is not None
    assert analysis_by_charge.physical_extrapolation.bunch_charge_pc == pytest.approx(
        100.0
    )

    with pytest.raises(ValueError):
        analyze_energy_transfer(
            total_energy.total_energy_au,
            u_inc,
            doppler_factor,
            num_electrons,
            physical_num_electrons=1e6,
            physical_bunch_charge_pc=1.0,
        )

    save_json(tmp_path / "energy_summary.json", {"analysis": analysis_by_count})
    assert (tmp_path / "energy_summary.json").exists()

    spec_summary = compute_detector_spectral_energy(
        grid.reshape_to_grid(np.asarray(E_ref)),
        grid.reshape_to_grid(np.asarray(B_ref)),
        grid.area_weights_2d,
        normal_vector=grid.normal_lab,
        detector_positions=grid.reshape_to_grid(grid.lab_positions_flat),
    )
    summary_dict = build_energy_summary_dict(
        analysis_by_count,
        detector_spectral_energy=spec_summary,
        detector_total_energy=total_energy,
        factorized_detector_energy=fact_energy,
        frequency_quadrature=quad,
        energy_method="multifreq-quadrature",
    )
    summary_path = tmp_path / "energy_summary_full.json"
    save_json(summary_path, summary_dict)
    assert summary_path.exists()
    text_json = summary_path.read_text()
    assert "spectral_energy_flux" not in text_json
    assert "fluence_per_pixel" not in text_json
    assert "multifreq-quadrature" in text_json

    report = format_energy_transfer_report(
        analysis_by_count,
        detector_spectral_energy=spec_summary,
        detector_total_energy=total_energy,
        factorized_detector_energy=fact_energy,
    )
    assert "Detector energy (exact multi-freq)" in report
    assert "Thin-sheet theory" in report
    assert "Incoherent Larmor emission" in report
    assert "Physical bunch extrapolation" in report
