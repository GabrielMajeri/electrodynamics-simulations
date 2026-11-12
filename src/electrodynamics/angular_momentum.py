from typing import cast

import numpy as np

from electrodynamics.constants import SPEED_OF_LIGHT as c


def compute_angular_momentum_derivative(
    detector_positions: np.ndarray, electric_field: np.ndarray, omega: float
) -> np.ndarray:
    assert detector_positions.ndim == 3, (
        "Detector position should be a 2D array of 3D vectors"
    )
    assert detector_positions.shape[-1] == 3, (
        "Detector position should be an array of 3D vectors"
    )

    assert electric_field.ndim == 3, (
        "Electric field should be a 2D array of 3D complex phasors"
    )
    assert electric_field.shape[-1] == 3, (
        "Electric field should be an array of 3D complex phasors"
    )

    dx: float = (detector_positions[1, 0, 0] - detector_positions[0, 0, 0]).item()
    dy: float = (detector_positions[0, 1, 1] - detector_positions[0, 0, 1]).item()

    # Factor out the radially symmetric spherical propagation phase exp(i * (omega/c) * |r_det|)
    # before taking Cartesian finite differences. Since (x d/dy - y d/dx) |r_det| == 0 analytically,
    # this is an exact identity that eliminates Cartesian grid dispersion (sin(4*phi) aliasing).
    r_det = np.linalg.norm(detector_positions, axis=-1, keepdims=True)
    r_center = float(
        r_det[detector_positions.shape[0] // 2, detector_positions.shape[1] // 2, 0]
    )
    spherical_phase = np.exp(-1j * (omega / c) * (r_det - r_center))
    smooth_field = electric_field * spherical_phase

    electric_field_dx, electric_field_dy = cast(
        tuple[np.ndarray, np.ndarray],
        np.gradient(smooth_field[:, :, :], dx, dy, axis=(0, 1)),
    )

    dL_z = np.zeros(detector_positions.shape[:2], dtype=np.complex128)

    for i in range(3):
        dL_z += (
            smooth_field[:, :, i]
            / (1j * omega)
            * (
                detector_positions[:, :, 0] * np.conjugate(electric_field_dy[:, :, i])
                - detector_positions[:, :, 1] * np.conjugate(electric_field_dx[:, :, i])
            )
        )

    dL_z /= 4 * np.pi

    return dL_z
