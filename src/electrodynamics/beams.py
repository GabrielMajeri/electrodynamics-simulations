from math import pi

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from electrodynamics.constants import SPEED_OF_LIGHT as c


@jdc.pytree_dataclass
class Polarization:
    x: complex
    y: complex

    def __init__(self, x: jdc.Static[complex], y: jdc.Static[complex]) -> None:
        norm = abs(x) + abs(y)
        if not jnp.isclose(norm, 1):
            raise ValueError("Polarization should have unit norm")

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


@jdc.pytree_dataclass
class LaserParameters:
    frequency: float
    wavelength: float
    amplitude: float
    polarization: Polarization


@jdc.pytree_dataclass
class GaussianBeamParameters(LaserParameters):
    waist_radius: float


@jdc.pytree_dataclass
class LaguerreGaussBeamParameters(GaussianBeamParameters):
    radial_index: int
    azimuthal_index: int


@jdc.jit
def compute_plane_wave_fields(
    parameters: jdc.Static[LaserParameters], position: jax.Array
) -> tuple[jax.Array, jax.Array]:
    assert position.shape[-1] == 4, "Positions must be 4-vectors"

    tc, _, _, z = position.T
    t = tc / c

    # k
    wavenumber = (2 * pi) / parameters.wavelength

    magnitude = parameters.amplitude
    phase = jnp.exp(1j * (parameters.frequency * t - wavenumber * z))

    phasor = magnitude * phase

    polarization = parameters.polarization

    E_x = jnp.real(polarization.x * phasor)
    E_y = jnp.real(polarization.y * phasor)
    E_z = jnp.zeros_like(E_x)

    E = jnp.vstack((E_x, E_y, E_z)).T

    B_x = -E_y / c
    B_y = E_x / c
    B_z = jnp.zeros_like(B_x)

    B = jnp.vstack((B_x, B_y, B_z)).T

    return E, B


@jdc.jit
def compute_gaussian_beam_fields(
    parameters: jdc.Static[GaussianBeamParameters], position: jax.Array
) -> tuple[jax.Array, jax.Array]:
    assert position.shape[-1] == 4, "Positions must be 4-vectors"

    tc, x, y, z = position.T
    t = tc / c

    r = jnp.hypot(x, y)

    # w_0 = w(0)
    refractive_index = 1.0
    rayleigh_range = (
        pi * parameters.waist_radius**2 * refractive_index
    ) / parameters.wavelength
    w_z = parameters.waist_radius * jnp.sqrt(1 + (z / rayleigh_range) ** 2)

    # k
    wave_number = (2 * pi * refractive_index) / parameters.wavelength
    # R(z)
    zero_z = jnp.isclose(z, 0)
    curvature_radius = jnp.where(
        zero_z, jnp.inf, z * (1 + jnp.square(rayleigh_range / jnp.where(zero_z, 1, z)))
    )

    gouy_phase = jnp.arctan(z / rayleigh_range)

    coefficients = (
        parameters.amplitude
        * (parameters.waist_radius / w_z)
        * jnp.exp(-(r**2) / (w_z**2))
        * jnp.exp(
            -1j
            * (
                wave_number * z
                + wave_number * (r**2) / (2 * curvature_radius)
                - gouy_phase
            )
            + 1j * parameters.frequency * t
        )
    )

    E_x = parameters.polarization.x * coefficients
    E_y = parameters.polarization.y * coefficients
    # TODO: better approximation formula for computing derivatives in the paraxial approximation
    E_z = (2j) / (wave_number * w_z**2) * (x * E_x + y * E_y)

    E = jnp.vstack((jnp.real(E_x), jnp.real(E_y), jnp.real(E_z))).T

    B_x = -E_y / c
    B_y = E_x / c
    B_z = 1j / (parameters.frequency * w_z**2) * (y * E_x - x * E_y)

    B = jnp.vstack((jnp.real(B_x), jnp.real(B_y), jnp.real(B_z))).T

    return E, B


@jdc.jit
def compute_laguerre_gauss_beam_fields(
    parameters: jdc.Static[LaguerreGaussBeamParameters], position: jax.Array
) -> tuple[jax.Array, jax.Array]:
    assert position.shape[-1] == 4, "Positions must be 4-vectors"

    tc, x, y, z = position.T
    t = tc / c

    r = jnp.hypot(x, y)
    phi = jnp.arctan2(y, x)

    # w_0 = w(0)
    rayleigh_length = (pi * parameters.waist_radius**2) / parameters.wavelength
    # FWHM
    w_z = parameters.waist_radius * jnp.sqrt(1 + (z / rayleigh_length) ** 2)

    # k
    wave_number = (2 * pi) / parameters.wavelength

    # R(z)
    small_z = jnp.abs(z) < 1e-8
    safe_z = jnp.where(small_z, 1, z)
    curvature_radius = jnp.where(
        small_z, 0.0, z * (1 + jnp.square(rayleigh_length / safe_z))
    )

    safe_curvature_radius = jnp.where(small_z, 1, curvature_radius)
    curvature_term = jnp.where(small_z, 0, (r**2) / (2 * safe_curvature_radius))

    gouy_phase = jnp.arctan(z / rayleigh_length)

    r_over_wz_all_squared = (r / w_z) ** 2

    coefficients = (
        parameters.amplitude
        * (parameters.waist_radius / w_z)
        * jnp.pow(jnp.sqrt(2) * r / w_z, abs(parameters.azimuthal_index))
        * laguerre_polynomial(
            parameters.radial_index,
            abs(parameters.azimuthal_index),
            2 * r_over_wz_all_squared,
        )
        * jnp.exp(-r_over_wz_all_squared)
        * jnp.exp(
            -1j
            * (
                wave_number * z
                + wave_number * curvature_term
                + parameters.azimuthal_index * phi
                - (2 * parameters.radial_index + abs(parameters.azimuthal_index) + 1)
                * gouy_phase
            )
            + 1j * parameters.frequency * t
        )
    )

    E_x = parameters.polarization.x * coefficients
    E_y = parameters.polarization.y * coefficients
    # TODO: better approximation formula for computing derivatives in the paraxial approximation
    E_z = (2j) / (wave_number * w_z**2) * (x * E_x + y * E_y)

    E = jnp.vstack((jnp.real(E_x), jnp.real(E_y), jnp.real(E_z))).T

    B_x = -E_y / c
    B_y = E_x / c
    B_z = 1j / (parameters.frequency * w_z**2) * (y * E_x - x * E_y)

    B = jnp.vstack((jnp.real(B_x), jnp.real(B_y), jnp.real(B_z))).T

    return E, B


@jdc.jit
def laguerre_polynomial(
    n: jdc.Static[int], alpha: jdc.Static[float], x: float | jax.Array
) -> float | jax.Array:
    if n == 0:
        return 1

    if n == 1:
        return 1 + alpha - x

    if n == 2:
        return 0.5 * (jnp.pow(x, 2) - 2 * (alpha + 2) * x + (alpha + 1) * (alpha + 2))

    return (
        (2 * n - 1 + alpha - x) * laguerre_polynomial(n - 1, alpha, x)
        - (n - 1 + alpha) * laguerre_polynomial(n - 2, alpha, x)
    ) / n
