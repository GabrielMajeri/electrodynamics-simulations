import math

import jax.numpy as jnp
import numpy as np

from electrodynamics.beams import (
    GaussianBeamParameters,
    LaguerreGaussBeamParameters,
    compute_gaussian_beam_fields,
    compute_laguerre_gauss_beam_fields,
    laguerre_polynomial,
    laguerre_polynomial_derivative,
)
from electrodynamics.constants import (
    SPEED_OF_LIGHT as c,
)
from electrodynamics.polarization import Polarizations


def test_beam_fields(jax_config) -> None:
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
    electric_field, magnetic_field = compute_laguerre_gauss_beam_fields(
        parameters, position
    )

    assert electric_field.shape == (1, 3)
    assert magnetic_field.shape == (1, 3)

    # Transverse relationship |B_perp| ~ |E_perp| / c
    E_perp = jnp.hypot(electric_field[0, 0], electric_field[0, 1])
    B_perp = jnp.hypot(magnetic_field[0, 0], magnetic_field[0, 1])
    np.testing.assert_allclose(B_perp, E_perp / c, rtol=1e-5)


def test_laguerre_polynomial_derivative() -> None:
    eps = 1e-4
    for n in [0, 1, 2, 3]:
        for alpha in [0.0, 1.0, 2.0]:
            for x in [0.1, 0.5, 2.0]:
                analytical = laguerre_polynomial_derivative(n, alpha, x)
                finite_difference = (
                    laguerre_polynomial(n, alpha, x + eps)
                    - laguerre_polynomial(n, alpha, x - eps)
                ) / (2 * eps)
                np.testing.assert_allclose(
                    analytical, finite_difference, rtol=1e-3, atol=1e-3
                )


def test_paraxial_exact_vs_approximate_options(jax_config) -> None:
    freq = 0.057
    wavelength = (2 * math.pi * c) / freq
    waist = 50 * wavelength

    # Test Gaussian beam options
    params_gauss_exact = GaussianBeamParameters(
        freq, wavelength, 1.0, Polarizations.LINEAR.value, waist, exact_paraxial=True
    )
    params_gauss_approx = GaussianBeamParameters(
        freq, wavelength, 1.0, Polarizations.LINEAR.value, waist, exact_paraxial=False
    )

    # Test Gaussian beam options at Rayleigh distance where curvature is maximal
    rayleigh = (math.pi * waist**2) / wavelength
    pos = jnp.array([[0.0, 10.0, 10.0, rayleigh]])
    em_exact = compute_gaussian_beam_fields(params_gauss_exact, pos)
    em_approx = compute_gaussian_beam_fields(params_gauss_approx, pos)

    # At z = z_R, curvature term causes a significant difference between exact and approximate
    assert not np.isclose(
        em_exact.electric_field[0, 2], em_approx.electric_field[0, 2], rtol=0.1
    )

    # Test LG beam options
    params_lg_exact = LaguerreGaussBeamParameters(
        freq,
        wavelength,
        1.0,
        Polarizations.RIGHT_CIRCULAR.value,
        waist,
        0,
        1,
        exact_paraxial=True,
    )
    params_lg_approx = LaguerreGaussBeamParameters(
        freq,
        wavelength,
        1.0,
        Polarizations.RIGHT_CIRCULAR.value,
        waist,
        0,
        1,
        exact_paraxial=False,
    )

    t_c = (math.pi / 2) * c / freq
    pos_origin = jnp.array([[t_c, 0.0, 0.0, 0.0]])
    em_lg_exact = compute_laguerre_gauss_beam_fields(params_lg_exact, pos_origin)
    em_lg_approx = compute_laguerre_gauss_beam_fields(params_lg_approx, pos_origin)

    # In approximate mode, Ez at origin vanishes:
    np.testing.assert_allclose(em_lg_approx.electric_field[0, 2], 0.0, atol=1e-12)
    # In exact mode, Ez at origin does NOT vanish for |l| = 1 with circular polarization:
    assert abs(em_lg_exact.electric_field[0, 2]) > 1e-4


def test_lg_exact_on_axis_field_hu_paper(jax_config) -> None:
    """Verifies that for LG(p=0, l=1) with right circular polarization,
    the on-axis `E_z` at `z = 0`, `t = pi/(2*omega)` matches Eq. (3) of Hu et al. (2018).
    In Eq. (3): at x=0, r=0, |Ex| = 2*sqrt(2)*E_L0 / (k * waist).
    """
    freq = 0.057
    wavelength = (2 * math.pi * c) / freq
    amp = 1.0
    waist = 3.0 * wavelength
    k = (2 * math.pi) / wavelength

    params_exact = LaguerreGaussBeamParameters(
        freq,
        wavelength,
        amp,
        Polarizations.RIGHT_CIRCULAR.value,
        waist,
        0,
        1,
        exact_paraxial=True,
    )
    t_c = (math.pi / 2) * c / freq
    pos_origin = jnp.array([[t_c, 0.0, 0.0, 0.0]])
    em = compute_laguerre_gauss_beam_fields(params_exact, pos_origin)

    Ez_origin = em.electric_field[0, 2].item()
    # Theoretical magnitude on axis: 2 * sqrt(2) * amp / (k * waist)
    expected_mag = (2.0 * math.sqrt(2.0) * amp) / (k * waist)
    np.testing.assert_allclose(abs(Ez_origin), expected_mag, rtol=1e-5)
