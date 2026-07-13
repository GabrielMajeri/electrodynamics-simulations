from enum import Enum

import numpy as np

import jax_dataclasses as jdc


@jdc.pytree_dataclass
class Polarization:
    x: complex
    y: complex

    def __init__(self, x: jdc.Static[complex], y: jdc.Static[complex]) -> None:
        norm = abs(x) ** 2 + abs(y) ** 2
        if not np.isclose(norm, 1):
            raise ValueError("Polarization should have unit norm")

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


class Polarizations(Enum):
    LINEAR = Polarization(1.0, 0.0)
    RIGHT_CIRCULAR = Polarization(1.0 / np.sqrt(2).item(), 1j / np.sqrt(2).item())
    LEFT_CIRCULAR = Polarization(1.0 / np.sqrt(2).item(), -1j / np.sqrt(2).item())
