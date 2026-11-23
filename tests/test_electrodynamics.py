from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from electrodynamics.beams import LaguerreGaussBeamParameters
from electrodynamics.constants import (
    ELECTRON_CHARGE as q,
)
from electrodynamics.constants import (
    ELECTRON_MASS as m_e,
)
from electrodynamics.constants import (
    SPEED_OF_LIGHT as c,
)
from electrodynamics.fields import (
    compute_coherent_detector_and_spectrum_fields,
    compute_doppler_shift_factor,
    compute_relativistic_reflection_parameters,
    compute_scattered_electric_and_magnetic_fields,
)
from electrodynamics.integrate import (
    integration_step_rk4_lab_time,
    parse_snapshot_schedule,
)
from electrodynamics.polarization import (
    Polarizations,
    get_polarization_by_name,
    resolve_circular_polarization,
)
from electrodynamics.pulse import (
    CosineSquaredPulseParameters,
    cosine_squared_envelope,
)
from electrodynamics.serialization import prepare_output_directory, save_json


def test_constants() -> None:
    assert c == 137.036
    assert m_e == 1.0
    assert q == -1.0


def test_doppler_shift() -> None:
    gamma = 5.0
    a_0 = 0.1
    doppler = compute_doppler_shift_factor(gamma, m_e, q, a_0)
    # Doppler factor for counter-propagating head-on collision is ~ 4 * gamma^2 when a_0 << 1
    assert doppler > 1.0
    assert 4 * (gamma**2) * 0.8 < doppler < 4 * (gamma**2) * 1.1


def test_scattered_fields(jax_config) -> None:
    frequency = 0.057
    position = jnp.array([[0.0, 10.0, 10.0, 0.0]])
    momentum = jnp.array([[c, 0.0, 0.0, 0.0]])
    initial_position = jnp.array([[0.0, 10.0, 10.0, 0.0]])
    detector_position = jnp.array([0.0, 0.0, -1000.0])

    E, B = compute_scattered_electric_and_magnetic_fields(
        frequency,
        position,
        momentum,
        initial_position,
        detector_position,
        include_second_order_terms=False,
    )
    assert E.shape == (1, 3)
    assert B.shape == (1, 3)
    assert jnp.all(jnp.isfinite(E))
    assert jnp.all(jnp.isfinite(B))


def test_cosine_squared_pulse() -> None:
    tau = 20.0
    phi_0 = 50.0
    p = CosineSquaredPulseParameters(phi_0=phi_0, tau_0=tau)

    # Peak at phi = phi_0
    peak_val = cosine_squared_envelope(jnp.array([phi_0]), p.phi_0, p.tau_0)
    np.testing.assert_allclose(peak_val, 1.0, atol=1e-6)

    # Edge at phi = phi_0 +/- tau
    edge_val1 = cosine_squared_envelope(jnp.array([phi_0 + tau]), p.phi_0, p.tau_0)
    edge_val2 = cosine_squared_envelope(jnp.array([phi_0 - tau]), p.phi_0, p.tau_0)
    np.testing.assert_allclose(edge_val1, 0.0, atol=1e-6)
    np.testing.assert_allclose(edge_val2, 0.0, atol=1e-6)

    # Outside pulse
    out_val = cosine_squared_envelope(jnp.array([phi_0 + 2 * tau]), p.phi_0, p.tau_0)
    np.testing.assert_allclose(out_val, 0.0, atol=1e-6)


def test_lab_time_rk4_integration() -> None:
    freq = 0.057
    wl = (2 * jnp.pi * c) / freq
    laser = LaguerreGaussBeamParameters(
        freq, wl, 1.0 * c * freq, Polarizations.RIGHT_CIRCULAR.value, 3.0 * wl, 0, 1
    )
    pulse = CosineSquaredPulseParameters(phi_0=0.0, tau_0=100.0)

    # Start at rest
    pos = jnp.array([[0.0, 100.0, 100.0, 0.0]])
    mom = jnp.array([[c, 0.0, 0.0, 0.0]])

    dt = 1.0
    for _ in range(50):
        pos, mom = integration_step_rk4_lab_time(pos, mom, dt, laser, pulse)

    # Invariant mass-shell condition: u_0^2 - |u|^2 = c^2
    norm_sq = mom[0, 0] ** 2 - jnp.sum(mom[0, 1:] ** 2)
    np.testing.assert_allclose(norm_sq, c**2, rtol=1e-8)


def test_relativistic_reflection_parameters_chen_2022() -> None:
    """Verifies the Lorentz transformation reflection angle, Doppler shift, and profile
    tilt angle against Chen et al., Scientific Reports 12:12524 (2022) for `theta_in = pi/6`, `beta = 0.5`.
    """
    beta = 0.5
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    theta_in = np.pi / 6.0
    _, theta_re, doppler_factor, profile_tilt = (
        compute_relativistic_reflection_parameters(gamma, theta_in)
    )

    # Paper reports theta_re = 0.176 rad (rounded from 0.178 rad),
    # lambda_1 = 0.281 - 0.283 um, phi = 0.403 rad
    np.testing.assert_allclose(theta_re, 0.17816, atol=1e-3)
    lambda_1_um = 0.8 / doppler_factor
    np.testing.assert_allclose(lambda_1_um, 0.2835, atol=3e-3)
    np.testing.assert_allclose(profile_tilt, 0.4046, atol=2e-3)

    # Head-on reflection (theta_in = 0): doppler_factor = (1 + beta) / (1 - beta) = 3.0
    _, theta_re_0, doppler_0, tilt_0 = compute_relativistic_reflection_parameters(
        gamma, 0.0
    )
    assert theta_re_0 == 0.0
    np.testing.assert_allclose(doppler_0, 3.0, rtol=1e-12)
    assert tilt_0 == 0.0


def test_scattered_fields_unaccelerated_relativistic_particle_zero(jax_config) -> None:
    """Verifies that a relativistic particle moving at constant velocity (u == u_init)
    produces identically zero scattered radiation when initial_momentum is provided.
    """
    frequency = 0.171
    gamma = 2.0
    uz = -np.sqrt(gamma**2 - 1.0) * c
    position = jnp.array([[100.0, 5.0, -3.0, -50.0]])
    momentum = jnp.array([[gamma * c, 0.0, 0.0, uz]])
    initial_position = jnp.array([[0.0, 5.0, -3.0, 0.0]])
    detector_position = jnp.array([20.0, 15.0, -5000.0])

    E_lab, B_lab = compute_scattered_electric_and_magnetic_fields(
        frequency,
        position,
        momentum,
        initial_position,
        detector_position,
        is_lab_time=True,
        initial_momentum=momentum,
    )
    np.testing.assert_allclose(E_lab, 0.0, atol=1e-15)
    np.testing.assert_allclose(B_lab, 0.0, atol=1e-15)

    E_prop, B_prop = compute_scattered_electric_and_magnetic_fields(
        frequency,
        position,
        momentum,
        initial_position,
        detector_position,
        is_lab_time=False,
        initial_momentum=momentum,
    )
    np.testing.assert_allclose(E_prop, 0.0, atol=1e-15)
    np.testing.assert_allclose(B_prop, 0.0, atol=1e-15)


def test_polarization_helpers() -> None:
    pol_lin, sig_lin = get_polarization_by_name("linear-x")
    assert pol_lin == Polarizations.LINEAR_X.value
    assert sig_lin == 0

    pol_ly, sig_ly = get_polarization_by_name("linear-y")
    assert pol_ly.x == 0.0 and pol_ly.y == 1.0
    assert sig_ly == 0

    pol_opp, sig_opp = resolve_circular_polarization(
        azimuthal_index=1, handedness="opposite"
    )
    assert pol_opp == Polarizations.RIGHT_CIRCULAR.value
    assert sig_opp == -1

    pol_same, sig_same = resolve_circular_polarization(
        azimuthal_index=1, handedness="same"
    )
    assert pol_same == Polarizations.LEFT_CIRCULAR.value
    assert sig_same == 1


def test_snapshot_schedule_and_serialization(tmp_path: Path) -> None:
    parsed, schedule = parse_snapshot_schedule(
        "20.0, 10.0, 40.0", laser_period=10.0, t_final_periods=35.0
    )
    assert parsed == [10.0, 20.0, 40.0]
    np.testing.assert_allclose(schedule, [100.0, 200.0, 350.0])

    dt = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    out_dir = prepare_output_directory("study", dt, overwrite=True, runs_root=tmp_path)
    assert out_dir == tmp_path / "study" / "latest"
    assert out_dir.is_dir()

    custom_dir = prepare_output_directory(
        "study", dt, run_tag="custom", runs_root=tmp_path
    )
    assert custom_dir == tmp_path / "study" / "custom"
    assert custom_dir.is_dir()

    ts_dir = prepare_output_directory("study", dt, overwrite=False, runs_root=tmp_path)
    assert ts_dir == tmp_path / "study" / "2026-01-02_03-04-05"
    assert ts_dir.is_dir()

    save_json(
        out_dir / "cfg.json", {"arr": np.array([1.0, 2.0]), "scalar": np.float64(3.5)}
    )
    assert (out_dir / "cfg.json").exists()


def test_coherent_detector_and_spectrum_helper(jax_config) -> None:
    pos = jnp.array([[0.0, 1.0, 2.0, 0.0], [0.0, -1.0, 0.5, 0.0]])
    mom = jnp.array([[c, 0.1, 0.0, 0.0], [c, -0.1, 0.2, 0.0]])
    orig = jnp.array([[0.0, 1.0, 2.0, 0.0], [0.0, -1.0, 0.5, 0.0]])
    det_pos = jnp.array([[0.0, 0.0, -500.0], [10.0, 0.0, -500.0]])
    spec_pos = jnp.array([5.0, 0.0, -500.0])
    freqs = jnp.array([0.05, 0.06, 0.07])

    dE, dB, dE_spec = compute_coherent_detector_and_spectrum_fields(
        pos, mom, orig, det_pos, 0.057, spec_pos, freqs, is_lab_time=True
    )
    assert dE.shape == (2, 3)
    assert dB.shape == (2, 3)
    assert dE_spec.shape == (3, 3)
