import numpy as np

from electrodynamics.constants import (
    ELECTRON_MASS as m_e,
)
from electrodynamics.constants import (
    SPEED_OF_LIGHT as c,
)
from electrodynamics.initial_conditions import (
    generate_initial_particle_momenta_moving_towards_laser,
    generate_initial_positions_uniformly_on_disk,
    generate_initial_positions_uniformly_within_ball,
)


def test_initial_conditions_disk():
    rng = np.random.default_rng(123)
    radius = 50.0
    n = 2000
    pos = generate_initial_positions_uniformly_on_disk(rng, radius, n)

    assert pos.shape == (n, 3)
    # Check z coordinates are zero
    np.testing.assert_allclose(pos[:, 2], 0.0)
    # Check all points lie within disk
    radii = np.hypot(pos[:, 0], pos[:, 1])
    assert np.all(radii <= radius)
    # Mean should be close to zero
    assert abs(pos[:, 0].mean()) < 5.0
    assert abs(pos[:, 1].mean()) < 5.0


def test_initial_conditions_ball():
    rng = np.random.default_rng(123)
    radius = 30.0
    n = 1500
    pos = generate_initial_positions_uniformly_within_ball(rng, radius, n)

    assert pos.shape == (n, 3)
    radii = np.linalg.norm(pos, axis=-1)
    assert np.all(radii <= radius + 1e-12)


def test_initial_momenta_energy_momentum_relation():
    gamma = 2.5
    n = 100
    momenta = generate_initial_particle_momenta_moving_towards_laser(n, gamma, m_e)

    assert momenta.shape == (n, 4)
    # u^0 = gamma * m * c
    np.testing.assert_allclose(momenta[:, 0], gamma * m_e * c)
    # u^3 = -sqrt(gamma^2 - 1) * m * c
    np.testing.assert_allclose(momenta[:, 3], -np.sqrt(gamma**2 - 1) * m_e * c)
    # 4-momentum norm: u_0^2 - |u|^2 = (m * c)^2
    norm_sq = momenta[:, 0] ** 2 - np.sum(momenta[:, 1:] ** 2, axis=-1)
    np.testing.assert_allclose(norm_sq, (m_e * c) ** 2, rtol=1e-10)
