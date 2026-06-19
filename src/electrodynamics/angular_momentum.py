from typing import cast

import numpy as np

from .typing import ComplexArray, RealArray


def compute_angular_momentum_derivative(
    detector_positions: RealArray, electric_field: ComplexArray, omega: float
) -> ComplexArray:
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

    electric_field_dx, electric_field_dy = cast(
        tuple[ComplexArray, ComplexArray],
        np.gradient(electric_field[:, :, :], dx, dy, axis=(0, 1)),
    )

    dL_z = np.zeros(detector_positions.shape[:2], dtype=np.complex128)

    for i in range(3):
        dL_z += (
            electric_field[:, :, i]
            / (1j * omega)
            * (
                detector_positions[:, :, 0] * np.conjugate(electric_field_dy[:, :, i])
                - detector_positions[:, :, 1] * np.conjugate(electric_field_dx[:, :, i])
            )
        )

    dL_z /= 4 * np.pi

    return dL_z
