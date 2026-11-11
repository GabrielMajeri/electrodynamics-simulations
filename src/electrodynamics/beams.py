from math import factorial, pi, sqrt
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from .constants import SPEED_OF_LIGHT as c
from .polarization import Polarization


class EMField(NamedTuple):
    "Convenience wrapper for the electric and magnetic field vectors."

    electric_field: jax.Array
    magnetic_field: jax.Array


@jdc.pytree_dataclass
class LaserParameters:
    "Basic parameters for an abstract laser beam (e.g. plane wave)."

    frequency: float
    wavelength: float
    amplitude: float
    polarization: Polarization


@jdc.pytree_dataclass
class GaussianBeamParameters(LaserParameters):
    "Parameters for a Gaussian beam."

    waist_radius: float
    exact_paraxial: bool = True

    def __init__(
        self,
        frequency: float,
        wavelength: float,
        amplitude: float,
        polarization: Polarization,
        waist_radius: float,
        exact_paraxial: bool = True,
    ) -> None:
        # Use `object.__setattr__` since data classes are "frozen"
        # and regular attribute assignment doesn't work.
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "waist_radius", waist_radius)
        object.__setattr__(self, "exact_paraxial", exact_paraxial)


@jdc.pytree_dataclass
class LaguerreGaussBeamParameters(GaussianBeamParameters):
    "Parameters for a Laguerre-Gauss beam."

    radial_index: int = 0
    azimuthal_index: int = 0
    exact_paraxial: bool = True

    def __init__(
        self,
        frequency: float,
        wavelength: float,
        amplitude: float,
        polarization: Polarization,
        waist_radius: float,
        radial_index: int = 0,
        azimuthal_index: int = 0,
        exact_paraxial: bool = True,
    ) -> None:
        super().__init__(
            frequency,
            wavelength,
            amplitude,
            polarization,
            waist_radius,
            exact_paraxial,
        )
        object.__setattr__(self, "radial_index", radial_index)
        object.__setattr__(self, "azimuthal_index", azimuthal_index)


@jdc.jit
def compute_plane_wave_fields(
    parameters: jdc.Static[LaserParameters], position: jax.Array
) -> EMField:
    """Evaluates the formulae for the electric and magnetic fields of
    the plane wave beam with the given parameters, at the given position(s).
    """
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

    return EMField(electric_field=E, magnetic_field=B)


@jdc.jit
def compute_gaussian_beam_fields(
    parameters: jdc.Static[GaussianBeamParameters],
    position: jax.Array,
    exact_paraxial: jdc.Static[bool] | None = None,
) -> EMField:
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

    B_x = -E_y / c
    B_y = E_x / c

    use_exact = parameters.exact_paraxial if exact_paraxial is None else exact_paraxial

    if not use_exact:
        # Fast approximation: differentiate only the Gaussian exp(-r^2/w_z^2) envelope
        E_z = (2j) / (wave_number * w_z**2) * (x * E_x + y * E_y)
        B_z = (2j) / (parameters.frequency * w_z**2) * (y * E_x - x * E_y)
    else:
        # Exact transverse derivatives including wavefront curvature:
        # d/dx u = -x * (2 / w_z^2 + i * k / R(z)) * u
        safe_R = jnp.where(zero_z, jnp.inf, curvature_radius)
        Q = 2.0 / w_z**2 + jnp.where(zero_z, 0.0, 1j * wave_number / safe_R)

        E_z = (1j * Q / wave_number) * (x * E_x + y * E_y)
        B_z = (1j * Q / parameters.frequency) * (y * E_x - x * E_y)

    E = jnp.vstack((jnp.real(E_x), jnp.real(E_y), jnp.real(E_z))).T
    B = jnp.vstack((jnp.real(B_x), jnp.real(B_y), jnp.real(B_z))).T

    return EMField(electric_field=E, magnetic_field=B)


@jdc.jit
def compute_laguerre_gauss_beam_fields(
    parameters: jdc.Static[LaguerreGaussBeamParameters],
    position: jax.Array,
    exact_paraxial: jdc.Static[bool] | None = None,
) -> EMField:
    assert position.shape[-1] == 4, "Positions must be 4-vectors"

    tc, x, y, z = position.T
    t = tc / c

    r = jnp.hypot(x, y)

    p = parameters.radial_index
    l = parameters.azimuthal_index
    abs_l = abs(l)

    normalization_factor = sqrt(2 * factorial(p) / factorial(p + abs_l))

    # w_0 = w(0)
    rayleigh_length = (pi * parameters.waist_radius**2) / parameters.wavelength
    # FWHM
    w_z = parameters.waist_radius * jnp.sqrt(1 + (z / rayleigh_length) ** 2)

    # k
    wave_number = (2 * pi) / parameters.wavelength

    # R(z)
    small_z = jnp.abs(z) < 1e-8
    safe_z = jnp.where(small_z, 1.0, z)
    curvature_radius = jnp.where(
        small_z, jnp.inf, z * (1 + jnp.square(rayleigh_length / safe_z))
    )

    safe_curvature_radius = jnp.where(small_z, 1.0, curvature_radius)
    curvature_term = jnp.where(small_z, 0.0, (r**2) / (2 * safe_curvature_radius))

    gouy_phase = jnp.arctan(z / rayleigh_length)

    r_over_wz_all_squared = (r / w_z) ** 2
    xi = 2 * r_over_wz_all_squared

    L_val = laguerre_polynomial(p, abs_l, xi)

    # Envelope factor without vortex polynomial
    phi_env = (
        normalization_factor
        * parameters.amplitude
        * (parameters.waist_radius / w_z)
        * jnp.exp(-r_over_wz_all_squared)
        * jnp.exp(
            -1j
            * (
                wave_number * z
                + wave_number * curvature_term
                - (2 * p + abs_l + 1) * gouy_phase
            )
            + 1j * parameters.frequency * t
        )
    )

    # Vortex factor: V(x, y) = [(sqrt(2)/w_z) * (x - i * sgn(l) * y)]^|l|
    # This Cartesian representation is singularity-free at r = 0 and identical to (sqrt(2)*r/w_z)^|l| * e^(-i*l*phi)
    s = 1 if l >= 0 else -1
    v_base = (x - 1j * s * y) * (jnp.sqrt(2) / w_z)

    if abs_l == 0:
        V_val = 1.0 + 0j
        dV_dx = 0.0 + 0j
    elif abs_l == 1:
        V_val = v_base
        dV_dx = (jnp.sqrt(2) / w_z) + 0j
    else:
        V_val = v_base**abs_l
        dV_dx = abs_l * (jnp.sqrt(2) / w_z) * (v_base ** (abs_l - 1))
    dV_dy = -1j * s * dV_dx

    coefficients = phi_env * V_val * L_val

    E_x = parameters.polarization.x * coefficients
    E_y = parameters.polarization.y * coefficients

    B_x = -E_y / c
    B_y = E_x / c

    use_exact = parameters.exact_paraxial if exact_paraxial is None else exact_paraxial

    if not use_exact:
        # Fast approximation: differentiate only the Gaussian exp(-r^2/w_z^2) envelope
        E_z = (2j) / (wave_number * w_z**2) * (x * E_x + y * E_y)
        B_z = (2j) / (parameters.frequency * w_z**2) * (y * E_x - x * E_y)
    else:
        # Exact transverse paraxial derivatives
        dL_val = laguerre_polynomial_derivative(p, abs_l, xi)
        Q = 2.0 / w_z**2 + jnp.where(
            small_z, 0.0, 1j * wave_number / safe_curvature_radius
        )

        T_env = phi_env * (Q * V_val * L_val - V_val * (4.0 / w_z**2) * dL_val)

        du_dx = phi_env * dV_dx * L_val - x * T_env
        du_dy = phi_env * dV_dy * L_val - y * T_env

        # E_z = -i / k * (eps_x * du/dx + eps_y * du/dy)
        E_z = (-1j / wave_number) * (
            parameters.polarization.x * du_dx + parameters.polarization.y * du_dy
        )
        # B_z = i / omega * (eps_y * du/dx - eps_x * du/dy)
        B_z = (1j / parameters.frequency) * (
            parameters.polarization.y * du_dx - parameters.polarization.x * du_dy
        )

    E = jnp.vstack((jnp.real(E_x), jnp.real(E_y), jnp.real(E_z))).T
    B = jnp.vstack((jnp.real(B_x), jnp.real(B_y), jnp.real(B_z))).T

    return EMField(electric_field=E, magnetic_field=B)


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


@jdc.jit
def laguerre_polynomial_derivative(
    n: jdc.Static[int], alpha: jdc.Static[float], x: float | jax.Array
) -> float | jax.Array:
    "Exact derivative d/dx L_{n}^{alpha(x)} = -L_{n-1}^{alpha+1}(x)."
    if n <= 0:
        return 0.0
    return -laguerre_polynomial(n - 1, alpha + 1, x)
