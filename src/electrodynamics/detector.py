import jax_dataclasses as jdc
import numpy as np
from scipy.spatial.transform import Rotation


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
    xs = np.linspace(
        -parameters.width / 2, parameters.width / 2, parameters.grid_size_x
    )
    ys = np.linspace(
        -parameters.height / 2, parameters.height / 2, parameters.grid_size_y
    )

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    grid_z = np.full_like(grid_x, z_distance)

    return np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)


def initialize_detector_positions(
    parameters: DetectorParameters, position: np.ndarray, view_direction: np.ndarray
) -> np.ndarray:
    """Initializes the positions for the 2D grid describing the simulated detector's pixels,
    located at an arbitrary point in space, looking towards the given direction.
    """
    xs = np.linspace(
        -parameters.width / 2, parameters.width / 2, parameters.grid_size_x
    )
    ys = np.linspace(
        -parameters.height / 2, parameters.height / 2, parameters.grid_size_y
    )

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    grid_z = np.zeros_like(grid_x)

    positions_arr = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)

    rotation = Rotation.from_rotvec(view_direction)
    positions_arr = rotation.apply(positions_arr)

    positions_arr += position

    return positions_arr


def initialize_detector_positions_reflected(
    parameters: DetectorParameters,
    distance: float,
    reflection_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Initializes the 3D laboratory positions and 2D local detector coordinates
    for a rectangular detector located at `distance` along the reflected beam axis
      k_re_hat = (sin(reflection_angle), 0, -cos(reflection_angle)),
    oriented perpendicular to k_re_hat.

    When `reflection_angle == 0.0`, this is identical to a detector at `z = -distance`
    parallel to the x-y plane.

    Returns:
        (lab_positions_flat, local_coords_2d)
        where lab_positions_flat has shape (Nx * Ny, 3) and
        local_coords_2d has shape (Nx, Ny, 3) with (x'_d, y'_d, -distance).
    """
    xs = np.linspace(
        -parameters.width / 2, parameters.width / 2, parameters.grid_size_x
    )
    ys = np.linspace(
        -parameters.height / 2, parameters.height / 2, parameters.grid_size_y
    )

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    grid_z = np.full_like(grid_x, -abs(distance))

    local_coords_2d = np.stack((grid_x, grid_y, grid_z), axis=-1)

    if np.isclose(reflection_angle, 0.0):
        return local_coords_2d.reshape(-1, 3), local_coords_2d

    cos_r = np.cos(reflection_angle)
    sin_r = np.sin(reflection_angle)
    dist = abs(distance)

    # Center of detector is at dist * (sin_r, 0, -cos_r)
    # Transverse axes on detector: e_x' = (cos_r, 0, sin_r), e_y' = (0, 1, 0)
    lab_x = dist * sin_r + grid_x * cos_r
    lab_y = grid_y
    lab_z = -dist * cos_r + grid_x * sin_r

    lab_positions_flat = np.stack((lab_x, lab_y, lab_z), axis=-1).reshape(-1, 3)
    return lab_positions_flat, local_coords_2d


def project_fields_to_detector_frame(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    reflection_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Projects 3D laboratory electric and magnetic field vectors onto the local
    coordinate frame of the reflected-beam detector:
      e_x' = (cos(theta_re), 0, sin(theta_re))
      e_y' = (0, 1, 0)
      e_z' = (-sin(theta_re), 0, cos(theta_re))
    """
    if np.isclose(reflection_angle, 0.0):
        return electric_field, magnetic_field

    cos_r = np.cos(reflection_angle)
    sin_r = np.sin(reflection_angle)

    E_xp = electric_field[..., 0] * cos_r + electric_field[..., 2] * sin_r
    E_yp = electric_field[..., 1]
    E_zp = -electric_field[..., 0] * sin_r + electric_field[..., 2] * cos_r

    B_xp = magnetic_field[..., 0] * cos_r + magnetic_field[..., 2] * sin_r
    B_yp = magnetic_field[..., 1]
    B_zp = -magnetic_field[..., 0] * sin_r + magnetic_field[..., 2] * cos_r

    return (
        np.stack((E_xp, E_yp, E_zp), axis=-1),
        np.stack((B_xp, B_yp, B_zp), axis=-1),
    )
