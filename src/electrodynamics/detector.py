import numpy as np
from scipy.spatial.transform import Rotation

import jax_dataclasses as jdc


@jdc.pytree_dataclass
class DetectorParameters:
    "Parameters for the simulated (rectangular) detector."

    width: float
    "Detector width (in atomic units)"

    height: float
    "Detector height (in atomic units)"

    grid_size_x: int
    "Number of points for discretizing the detector in the `x` direction"

    grid_size_y: int
    "Number of points for discretizing the detector in the `y` direction"


def initialize_detector_positions_negative_z(
    parameters: DetectorParameters,
    z_distance: float,
) -> np.ndarray:
    """Initializes the positions for the 2D grid describing the simulated detector's pixels,
    lying at some point along the z axis and being parallel to the x-y plane.
    """
    ys = np.linspace(
        -parameters.height / 2, parameters.height / 2, parameters.grid_size_y
    )
    xs = np.linspace(
        -parameters.width / 2, parameters.width / 2, parameters.grid_size_x
    )

    positions: list[tuple[float, float, float]] = []

    for y in ys:
        for x in xs:
            positions.append((x, y, z_distance))

    return np.asarray(positions)


def initialize_detector_positions(
    parameters: DetectorParameters, position: np.ndarray, view_direction: np.ndarray
) -> np.ndarray:
    """Initializes the positions for the 2D grid describing the simulated detector's pixels,
    located at an arbitrary point in space, looking towards the given direction.
    """
    # First, we generate the points at the origin
    ys = np.linspace(
        -parameters.height / 2, parameters.height / 2, parameters.grid_size_y
    )
    xs = np.linspace(
        -parameters.width / 2, parameters.width / 2, parameters.grid_size_x
    )

    positions_list: list[tuple[float, float, float]] = []

    for y in ys:
        for x in xs:
            positions_list.append((x, y, 0))

    positions_arr = np.asarray(positions_list)

    rotation = Rotation.from_rotvec(view_direction)
    positions_arr = rotation.apply(positions_arr)

    positions_arr += position

    return positions_arr
