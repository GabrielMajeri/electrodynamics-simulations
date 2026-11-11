import jax.numpy as jnp

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
