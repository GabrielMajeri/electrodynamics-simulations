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
    compute_doppler_shift_factor,
    compute_scattered_electric_and_magnetic_fields,
)
from electrodynamics.integrate import integration_step_rk4_lab_time
from electrodynamics.polarization import Polarizations
from electrodynamics.pulse import (
    CosineSquaredPulseParameters,
    cosine_squared_envelope,
)


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
