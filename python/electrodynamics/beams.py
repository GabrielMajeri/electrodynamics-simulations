from dataclasses import dataclass

from numba import njit
import numpy as np

from electrodynamics.constants import c

from .typing import ComplexArray, RealArray


@dataclass
class PolarizationVector:
    x: complex
    y: complex

    def __init__(self, x: complex, y: complex):
        if not np.isclose(abs(x) ** 2 + abs(y) ** 2, 1):
            raise ValueError("|x|^2 + |y|^2 must be approximately equal to 1")

        self.x = x
        self.y = y

    def to_numpy_array(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        return np.asarray((self.x, self.y), dtype=np.complex128)


def compute_gaussian_beam_electric_field(
    R: RealArray,
    Z: RealArray,
    amplitude: float,
    wavelength: float,
) -> ComplexArray:
    # w_0 = w(0)
    waist_radius = 2 * wavelength
    refractive_index = 1.0
    rayleigh_range = (np.pi * waist_radius**2 * refractive_index) / wavelength
    w_z = waist_radius * np.sqrt(1 + (Z / rayleigh_range) ** 2)

    # k
    wave_number = (2 * np.pi * refractive_index) / wavelength
    # R(z)
    curvature_radius = Z * (1 + (rayleigh_range / Z) ** 2)

    gouy_phase = np.arctan(Z / rayleigh_range)

    coefficients = (
        (waist_radius / w_z)
        * np.exp(-((R / w_z) ** 2))
        * np.exp(
            -1j
            * (
                wave_number * Z
                + wave_number * (R**2) / (2 * curvature_radius)
                - gouy_phase
            )
        )
    ).astype(np.complex128)

    return amplitude * coefficients


@njit(cache=True)
def compute_electric_and_magnetic_field_for_plane_wave(
    position: RealArray,
    time: RealArray,
    E: RealArray,
    B: RealArray,
) -> None:
    assert position.shape[-1] == 3, "Positions must be 3D vectors"

    z = position[2]

    E_0 = 1.0
    k = 1.0
    omega = 1.0

    # TODO: polarization

    E_x = E_0 * np.cos(k * z - omega * time)
    E_y = np.zeros_like(E_x)
    E_z = np.zeros_like(E_x)

    E[0] = E_x
    E[1] = E_y
    E[2] = E_z

    B_0 = E_0

    B_y = B_0 * np.cos(k * z - omega * time)
    B_x = np.zeros_like(B_y)
    B_z = np.zeros_like(B_y)

    B[0] = B_x
    B[1] = B_y
    B[2] = B_z


@njit(cache=True)
def compute_electric_and_magnetic_field_for_gaussian_beam(
    polarization: ComplexArray,
    position: RealArray,
    time: RealArray,
    E: RealArray,
    B: RealArray,
) -> None:
    assert position.shape[-1] == 3, "Positions must be 3D vectors"

    x = position[0]
    y = position[1]
    r = np.sqrt(np.square(x) + np.square(y))
    z = position[2]

    # E_0
    amplitude = 1.0

    # Frequency / angular speed
    omega = 1.0

    wavelength = 1

    # w_0 = w(0)
    waist_radius = 2 * wavelength
    refractive_index = 1.0
    rayleigh_range = (np.pi * waist_radius**2 * refractive_index) / wavelength
    w_z = waist_radius * np.sqrt(1 + (z / rayleigh_range) ** 2)

    # k
    wave_number = (2 * np.pi * refractive_index) / wavelength
    # R(z)
    zero_z = np.isclose(z, 0)
    curvature_radius = np.where(
        zero_z, np.inf, z * (1 + np.square(rayleigh_range / np.where(zero_z, 1, z)))
    )

    gouy_phase = np.arctan(z / rayleigh_range)

    coefficients = (
        (waist_radius / w_z)
        * np.exp(-(r**2) / (w_z**2))
        * np.exp(
            -1j
            * (
                wave_number * z
                + wave_number * (r**2) / (2 * curvature_radius)
                - gouy_phase
            )
            + 1j * omega * time
        )
    )

    E_x = polarization[0] * amplitude * coefficients
    E_y = polarization[1] * amplitude * coefficients
    # TODO: better approximation formula for computing derivatives in the paraxial approximation
    E_z = (2j) / (wave_number * w_z**2) * (x * E_x + y * E_y)

    E[0] = E_x.real
    E[1] = E_y.real
    E[2] = E_z.real

    B_x = -E_y / c
    B_y = E_x / c
    B_z = 1j / (omega * w_z**2) * (y * E_x - x * E_y)

    B[0] = B_x.real
    B[1] = B_y.real
    B[2] = B_z.real


@njit(cache=True)
def compute_electric_and_magnetic_field_for_laguerre_gauss_beam(
    # E_0
    amplitude: float,
    # w_0 = w(0)
    waist_radius: float,
    # Lambda
    wavelength: float,
    # p
    radial_index: int,
    # l (or m)
    azimuthal_index: int,
    # \xi
    polarization: ComplexArray,
    position: RealArray,
    time: float,
    E: RealArray,
    B: RealArray,
) -> None:
    assert position.shape[-1] == 3, "Position must be a 3D vector"

    x = position[0]
    y = position[1]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    z = position[2]

    # w_0 = w(0)
    rayleigh_length = (np.pi * waist_radius**2) / wavelength
    # FWHM
    w_z = waist_radius * np.sqrt(1 + (z / rayleigh_length) ** 2)

    omega = c * (2 * np.pi) / wavelength

    # k
    wave_number = (2 * np.pi) / wavelength
    # R(z)
    if abs(z) < 1e-8:
        curvature_radius = 0.0
    else:
        curvature_radius = z * (1 + np.square(rayleigh_length / z))

    if abs(curvature_radius) < 1e-5:
        curvature_term = 0
    else:
        curvature_term = (r**2) / (2 * curvature_radius)

    gouy_phase = np.arctan(z / rayleigh_length)

    r_over_wz_all_squared = (r / w_z) ** 2

    coefficients = (
        amplitude
        * (waist_radius / w_z)
        * np.pow(np.sqrt(2) * r / w_z, abs(azimuthal_index))
        * laguerre_polynomial(
            radial_index, abs(azimuthal_index), 2 * r_over_wz_all_squared
        )
        * np.exp(-r_over_wz_all_squared)
        * np.exp(
            -1j
            * (
                wave_number * z
                + wave_number * curvature_term
                + azimuthal_index * phi
                - (2 * radial_index + abs(azimuthal_index) + 1) * gouy_phase
            )
            + 1j * omega * time
        )
    )

    E_x = polarization[0] * coefficients
    E_y = polarization[1] * coefficients

    # TODO: better approximation formula for computing derivatives in the paraxial approximation
    E_z = (2j) / (wave_number * w_z**2) * (x * E_x + y * E_y)

    E[0] = E_x.real
    E[1] = E_y.real
    E[2] = E_z.real

    B_x = -E_y / c
    B_y = E_x / c
    B_z = 1j / (omega * w_z**2) * (y * E_x - x * E_y)

    B[0] = B_x.real
    B[1] = B_y.real
    B[2] = B_z.real


@njit(cache=True)
def laguerre_polynomial(n: int, alpha: float, x: float) -> float:
    if n == 0:
        return 1

    if n == 1:
        return 1 + alpha - x

    if n == 2:
        return 0.5 * (pow(x, 2) - 2 * (alpha + 2) * x + (alpha + 1) * (alpha + 2))

    return (
        (2 * n - 1 + alpha - x) * laguerre_polynomial(n - 1, alpha, x)
        - (n - 1 + alpha) * laguerre_polynomial(n - 2, alpha, x)
    ) / n
