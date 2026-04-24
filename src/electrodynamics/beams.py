from dataclasses import dataclass

import numpy as np
import scipy as sp

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


def compute_electric_and_magnetic_field_for_plane_wave(
    positions: RealArray, time: RealArray
) -> tuple[RealArray, RealArray]:
    assert positions.shape[-1] == 3, "Positions must be 3D vectors"

    z = positions[:, 2]

    E_0 = 1.0
    k = 1.0
    omega = 1.0

    # TODO: polarization

    E_x = E_0 * np.cos(k * z - omega * time)
    E_y = np.zeros_like(E_x)
    E_z = np.zeros_like(E_x)

    E = np.stack((E_x, E_y, E_z), axis=-1)

    B_0 = E_0

    B_y = B_0 * np.cos(k * z - omega * time)
    B_x = np.zeros_like(B_y)
    B_z = np.zeros_like(B_y)

    B = np.stack((B_x, B_y, B_z), axis=-1)

    # print("E =", E[0])
    # print("B =", B[0])

    return E, B


def compute_electric_and_magnetic_field_for_gaussian_beam(
    polarization: PolarizationVector, positions: RealArray, time: RealArray
) -> tuple[RealArray, RealArray]:
    assert positions.shape[-1] == 3, "Positions must be 3D vectors"

    x = positions[:, 0]
    y = positions[:, 1]
    r = np.sqrt(np.square(x) + np.square(y))
    z = positions[:, 2]

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

    E_x = polarization.x * amplitude * coefficients
    E_y = polarization.y * amplitude * coefficients
    # TODO: better approximation formula for computing derivatives in the paraxial approximation
    E_z = (2j) / (wave_number * w_z**2) * (x * E_x + y * E_y)

    E = np.stack((E_x, E_y, E_z), axis=-1).real.astype(np.float64)

    c = 1.0
    B_x = -E_y / c
    B_y = E_x / c
    B_z = 1j / (omega * w_z**2) * (y * E_x - x * E_y)

    B = np.stack((B_x, B_y, B_z), axis=-1).real.astype(np.float64)

    return E, B


def compute_electric_and_magnetic_field_for_laguerre_gauss_beam(
    amplitude: float,
    waist_radius: float,
    wavelength: float,
    radial_index: int,
    azimuthal_index: int,
    polarization: PolarizationVector,
    positions: RealArray,
    time: float | RealArray,
) -> tuple[RealArray, RealArray]:
    assert positions.shape[-1] == 3, "Positions must be 3D vectors"

    x = positions[:, 0]
    y = positions[:, 1]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    z = positions[:, 2]

    # w_0 = w(0)
    rayleigh_length = (np.pi * waist_radius**2) / wavelength
    # FWHM
    w_z = waist_radius * np.sqrt(1 + (z / rayleigh_length) ** 2)

    omega = c * (2 * np.pi) / wavelength

    # k
    wave_number = (2 * np.pi) / wavelength
    # R(z)
    zero_z = np.isclose(z, 0)
    curvature_radius = np.where(
        zero_z, np.inf, z * (1 + np.square(rayleigh_length / np.where(zero_z, 1, z)))
    )

    gouy_phase = np.arctan(z / rayleigh_length)

    r_over_wz_all_squared = (r / w_z) ** 2

    coefficients = (
        sp.special.poch((radial_index + 1), abs(azimuthal_index))
        * (waist_radius / w_z)
        * np.pow(np.sqrt(2) * r / w_z, abs(azimuthal_index))
        # TODO: use a faster approximation of 1F1
        * sp.special.hyp1f1(
            -radial_index, abs(azimuthal_index) + 1, 2 * r_over_wz_all_squared
        )
        * np.exp(-r_over_wz_all_squared)
        * np.exp(
            -1j
            * (
                wave_number * z
                + wave_number * (r**2) / (2 * curvature_radius)
                + azimuthal_index * phi
                - (2 * radial_index + abs(azimuthal_index) + 1) * gouy_phase
            )
            + 1j * omega * time
        )
    )

    E_x = polarization.x * amplitude * coefficients
    E_y = polarization.y * amplitude * coefficients

    # TODO: better approximation formula for computing derivatives in the paraxial approximation
    E_z = (2j) / (wave_number * w_z**2) * (x * E_x + y * E_y)

    E = np.real(np.stack((E_x, E_y, E_z), axis=-1)).astype(np.float64)

    B_x = -E_y / c
    B_y = E_x / c
    B_z = 1j / (omega * w_z**2) * (y * E_x - x * E_y)

    B = np.real(np.stack((B_x, B_y, B_z), axis=-1)).astype(np.float64)

    return E, B
