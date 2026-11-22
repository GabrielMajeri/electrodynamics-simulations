from enum import Enum

import jax_dataclasses as jdc
import numpy as np


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
    LINEAR_X = Polarization(1.0, 0.0)
    LINEAR_Y = Polarization(0.0, 1.0)
    RIGHT_CIRCULAR = Polarization(1.0 / np.sqrt(2).item(), 1j / np.sqrt(2).item())
    LEFT_CIRCULAR = Polarization(1.0 / np.sqrt(2).item(), -1j / np.sqrt(2).item())


_POLARIZATION_BY_NAME: dict[str, tuple[Polarization, int]] = {
    "linear-x": (Polarizations.LINEAR_X.value, 0),
    "linear-y": (Polarizations.LINEAR_Y.value, 0),
    "right-circular": (Polarizations.RIGHT_CIRCULAR.value, -1),
    "circular": (Polarizations.RIGHT_CIRCULAR.value, -1),
    "left-circular": (Polarizations.LEFT_CIRCULAR.value, 1),
}


def get_polarization_by_name(name: str) -> tuple[Polarization, int]:
    """Resolves a named polarization string to `(Polarization, spin_sigma)`."""
    if name not in _POLARIZATION_BY_NAME:
        raise ValueError(
            f"Unknown polarization '{name}'. Valid choices: {list(_POLARIZATION_BY_NAME.keys())}"
        )
    return _POLARIZATION_BY_NAME[name]


def resolve_circular_polarization(
    azimuthal_index: int, handedness: str = "opposite"
) -> tuple[Polarization, int]:
    """Resolves circular polarization and spin `sigma` relative to LG azimuthal index `l`.

    - `'opposite'`: spin-orbit cancellation (`sigma = -sgn(l)`).
    - `'same'`: co-rotating spin and OAM (`sigma = +sgn(l)`).
    """
    if handedness == "opposite":
        if azimuthal_index >= 0:
            return Polarizations.RIGHT_CIRCULAR.value, -1
        return Polarizations.LEFT_CIRCULAR.value, 1
    if handedness == "same":
        if azimuthal_index >= 0:
            return Polarizations.LEFT_CIRCULAR.value, 1
        return Polarizations.RIGHT_CIRCULAR.value, -1
    raise ValueError(
        f"Unknown polarization_handedness '{handedness}'. Choose 'opposite' or 'same'."
    )
