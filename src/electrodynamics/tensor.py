import numpy as np

from .typing import RealArray


def compute_electromagnetic_field_tensor(E: RealArray, B: RealArray) -> RealArray:
    zeros = np.zeros_like(E[:, 0])

    c = 1.0
    F = np.array(
        (
            (zeros, -E[:, 0] / c, -E[:, 1] / c, -E[:, 2] / c),
            (E[:, 0] / c, zeros, -B[:, 2], B[:, 1]),
            (E[:, 1] / c, B[:, 2], zeros, -B[:, 0]),
            (E[:, 2] / c, -B[:, 1], B[:, 0], zeros),
        ),
        dtype=np.float64,
    )

    # print("F =", F.T[0])

    return F.T
