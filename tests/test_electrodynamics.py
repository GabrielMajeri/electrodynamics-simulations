import math

import jax.numpy as jnp
import numpy as np
import pytest

from electrodynamics.beams import (
    LaguerreGaussBeamParameters,
    compute_laguerre_gauss_beam_fields,
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
    initialize_detector_positions,
    initialize_detector_positions_negative_z,
)
from electrodynamics.fields import (
    compute_doppler_shift_factor,
    compute_scattered_electric_and_magnetic_fields,
)
from electrodynamics.polarization import Polarizations


def test_constants() -> None:
    assert c == 137.036
    assert m_e == 1.0
    assert q == -1.0


def test_detector_positions_generation() -> None:
    params = DetectorParameters(width=100.0, height=80.0, grid_size_x=10, grid_size_y=8)
    z_dist = -500.0
    positions = initialize_detector_positions_negative_z(params, z_dist)

    assert positions.shape == (80, 3)
    np.testing.assert_allclose(positions[:, 2], z_dist)
    assert positions[:, 0].min() == pytest.approx(-50.0)
    assert positions[:, 0].max() == pytest.approx(50.0)
    assert positions[:, 1].min() == pytest.approx(-40.0)
    assert positions[:, 1].max() == pytest.approx(40.0)

    # General detector position generator
    pos_arbitrary = initialize_detector_positions(
        params, np.array([0.0, 0.0, z_dist]), np.zeros(3)
    )
    np.testing.assert_allclose(pos_arbitrary, positions)


def test_beam_fields() -> None:
    frequency = 0.057
    wavelength = (2 * math.pi * c) / frequency
    amplitude = 0.1
    waist = 50 * wavelength

    parameters = LaguerreGaussBeamParameters(
        frequency=frequency,
        wavelength=wavelength,
        amplitude=amplitude,
        polarization=Polarizations.RIGHT_CIRCULAR.value,
        waist_radius=waist,
        radial_index=0,
        azimuthal_index=0,
    )

    position = jnp.array([[0.0, 10.0, 10.0, 0.0]])
    em = compute_laguerre_gauss_beam_fields(parameters, position)

    assert em.electric_field.shape == (1, 3)
    assert em.magnetic_field.shape == (1, 3)

    # Transverse relationship |B_perp| ~ |E_perp| / c
    E_perp = jnp.hypot(em.electric_field[0, 0], em.electric_field[0, 1])
    B_perp = jnp.hypot(em.magnetic_field[0, 0], em.magnetic_field[0, 1])
    np.testing.assert_allclose(B_perp, E_perp / c, rtol=1e-5)


def test_doppler_shift() -> None:
    gamma = 5.0
    a_0 = 0.1
    doppler = compute_doppler_shift_factor(gamma, m_e, q, a_0)
    # Doppler factor for counter-propagating head-on collision is ~ 4 * gamma^2 when a_0 << 1
    assert doppler > 1.0
    assert 4 * (gamma**2) * 0.8 < doppler < 4 * (gamma**2) * 1.1


def test_scattered_fields() -> None:
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
