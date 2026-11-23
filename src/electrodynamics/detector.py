from dataclasses import dataclass
from typing import Literal, TypeVar

import jax
import jax_dataclasses as jdc
import numpy as np
from scipy.spatial.transform import Rotation

ArrayT = TypeVar("ArrayT", np.ndarray, jax.Array)
type DetectorShape = Literal["rect", "disk"]
type RadialSampling = Literal["gauss-legendre", "uniform", "equal-area"]


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

    def reshape_to_grid(self, array: ArrayT) -> ArrayT:
        """Reshapes a flattened `(grid_size_x * grid_size_y, ...)` detector array
        into 2D grid shape `(grid_size_x, grid_size_y, ...)`.
        """
        return array.reshape(self.grid_size_x, self.grid_size_y, *array.shape[1:])


@jdc.pytree_dataclass
class DiskDetectorParameters:
    "Parameters for a circular/disk detector discretized in polar coordinates `(r, phi)`."

    radius: float
    "Detector outer radius (in atomic units)"

    num_radial: int
    "Number of radial quadrature rings"

    num_azimuthal: int
    "Number of uniform azimuthal samples in [0, 2*pi)"

    radial_sampling: jdc.Static[RadialSampling] = "gauss-legendre"
    "Radial discretization scheme: 'gauss-legendre', 'uniform', or 'equal-area'"

    def reshape_to_grid(self, array: ArrayT) -> ArrayT:
        """Reshapes a flattened `(num_radial * num_azimuthal, ...)` detector array
        into 2D polar grid shape `(num_radial, num_azimuthal, ...)`.
        """
        return array.reshape(self.num_radial, self.num_azimuthal, *array.shape[1:])


type AnyDetectorParameters = DetectorParameters | DiskDetectorParameters


@dataclass(frozen=True)
class DetectorGrid:
    """Unified detector geometry, local coordinates, area quadrature weights,
    and outward surface normal vectors for rectangular or disk detectors.
    """

    parameters: AnyDetectorParameters
    lab_positions_flat: np.ndarray
    local_coords_2d: np.ndarray
    area_weights_2d: np.ndarray
    normal_lab: np.ndarray
    normal_local: np.ndarray

    @property
    def is_polar(self) -> bool:
        "Returns True if the detector uses polar `(r, phi)` disk discretization."
        return isinstance(self.parameters, DiskDetectorParameters)

    @property
    def grid_shape(self) -> tuple[int, int]:
        "Returns the 2D `(N1, N2)` shape of the detector grid."
        return (self.local_coords_2d.shape[0], self.local_coords_2d.shape[1])

    def reshape_to_grid(self, array: ArrayT) -> ArrayT:
        "Reshapes a flattened `(N1 * N2, ...)` array into `(N1, N2, ...)`."
        return self.parameters.reshape_to_grid(array)


def create_detector_parameters(
    shape: DetectorShape,
    span: float,
    grid_size: int = 64,
    num_radial: int | None = None,
    num_azimuthal: int | None = None,
    radial_sampling: RadialSampling = "gauss-legendre",
) -> AnyDetectorParameters:
    """Creates `DetectorParameters` (square) or `DiskDetectorParameters` from a common
    transverse span (width/height for `'rect'`, diameter `2 * radius` for `'disk'`).
    """
    if span <= 0.0:
        raise ValueError(f"Detector span must be positive, got {span}")

    if shape == "rect":
        return DetectorParameters(
            width=span,
            height=span,
            grid_size_x=grid_size,
            grid_size_y=grid_size,
        )
    if shape == "disk":
        n_r = num_radial if num_radial is not None else max(8, grid_size // 2)
        n_phi = num_azimuthal if num_azimuthal is not None else grid_size
        return DiskDetectorParameters(
            radius=span / 2.0,
            num_radial=n_r,
            num_azimuthal=n_phi,
            radial_sampling=radial_sampling,
        )
    raise ValueError(f"Unknown detector shape '{shape}'. Choose 'rect' or 'disk'.")


def compute_disk_radial_quadrature(
    parameters: DiskDetectorParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes 1D radial nodes, Jacobian-included radial weights (`r * dr`),
    and uniform azimuthal angles for a disk detector.

    Returns:
        `(radii_1d, radial_weights_1d, angles_1d)` such that
        ``sum_i radial_weights_1d[i] * sum_j (2 * pi / N_phi) * f(r_i, phi_j)``
        approximates the area integral over the disk of radius `parameters.radius`.
    """
    if parameters.radius <= 0.0:
        raise ValueError(f"Disk radius must be positive, got {parameters.radius}")
    if parameters.num_radial < 1 or parameters.num_azimuthal < 1:
        raise ValueError(
            f"num_radial and num_azimuthal must be >= 1, got ({parameters.num_radial}, {parameters.num_azimuthal})"
        )

    n_r = parameters.num_radial
    n_phi = parameters.num_azimuthal
    radius = parameters.radius

    angles_1d = (2.0 * np.pi * np.arange(n_phi, dtype=np.float64)) / n_phi

    if parameters.radial_sampling == "gauss-legendre":
        xi, w_gl = np.polynomial.legendre.leggauss(n_r)
        radii_1d = 0.5 * radius * (xi + 1.0)
        radial_weights_1d = 0.5 * radius * radii_1d * w_gl
    elif parameters.radial_sampling == "uniform":
        dr = radius / n_r
        radii_1d = (np.arange(n_r, dtype=np.float64) + 0.5) * dr
        radial_weights_1d = radii_1d * dr
    elif parameters.radial_sampling == "equal-area":
        radii_1d = radius * np.sqrt((np.arange(n_r, dtype=np.float64) + 0.5) / n_r)
        radial_weights_1d = np.full(n_r, 0.5 * (radius**2) / n_r, dtype=np.float64)
    else:
        raise ValueError(
            f"Unknown radial_sampling '{parameters.radial_sampling}'. "
            "Choose 'gauss-legendre', 'uniform', or 'equal-area'."
        )

    return radii_1d, radial_weights_1d, angles_1d


def _trapezoid_weights_1d(length: float, num_points: int) -> np.ndarray:
    "Computes 1D composite trapezoidal quadrature weights across `[ -L/2, +L/2 ]`."
    if num_points < 1:
        raise ValueError(f"Grid size must be >= 1, got {num_points}")
    if num_points == 1:
        return np.array([length], dtype=np.float64)
    step = length / (num_points - 1)
    weights = np.full(num_points, step, dtype=np.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def compute_detector_area_weights(parameters: AnyDetectorParameters) -> np.ndarray:
    """Computes the 2D area quadrature weights `dA` matching the detector geometry."""
    if isinstance(parameters, DiskDetectorParameters):
        _, radial_weights_1d, _ = compute_disk_radial_quadrature(parameters)
        dphi = (2.0 * np.pi) / parameters.num_azimuthal
        return np.broadcast_to(
            radial_weights_1d[:, None] * dphi,
            (parameters.num_radial, parameters.num_azimuthal),
        ).copy()

    wx = _trapezoid_weights_1d(parameters.width, parameters.grid_size_x)
    wy = _trapezoid_weights_1d(parameters.height, parameters.grid_size_y)
    return np.outer(wx, wy)


def circular_aperture_mask(local_coords_2d: np.ndarray, radius: float) -> np.ndarray:
    """Returns a boolean mask selecting detector pixels within `hypot(x', y') <= radius`."""
    if radius < 0.0:
        raise ValueError(f"Aperture radius must be non-negative, got {radius}")
    return np.hypot(local_coords_2d[..., 0], local_coords_2d[..., 1]) <= radius


def _create_detector_xy_grid(
    parameters: AnyDetectorParameters,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(parameters, DiskDetectorParameters):
        radii_1d, _, angles_1d = compute_disk_radial_quadrature(parameters)
        r_grid, phi_grid = np.meshgrid(radii_1d, angles_1d, indexing="ij")
        return r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid)

    xs = np.linspace(
        -parameters.width / 2, parameters.width / 2, parameters.grid_size_x
    )
    ys = np.linspace(
        -parameters.height / 2, parameters.height / 2, parameters.grid_size_y
    )
    return np.meshgrid(xs, ys, indexing="ij")


def initialize_detector_positions_negative_z(
    parameters: AnyDetectorParameters,
    z_distance: float,
) -> np.ndarray:
    """Initializes the positions for the 2D grid describing the simulated detector's pixels,
    lying at some point along the z axis and being parallel to the x-y plane.
    """
    grid_x, grid_y = _create_detector_xy_grid(parameters)
    grid_z = np.full_like(grid_x, z_distance)

    return np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)


def initialize_detector_positions(
    parameters: AnyDetectorParameters, position: np.ndarray, view_direction: np.ndarray
) -> np.ndarray:
    """Initializes the positions for the 2D grid describing the simulated detector's pixels,
    located at an arbitrary point in space, looking towards the given direction.
    """
    grid_x, grid_y = _create_detector_xy_grid(parameters)
    grid_z = np.zeros_like(grid_x)

    positions_arr = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)

    rotation = Rotation.from_rotvec(view_direction)
    positions_arr = rotation.apply(positions_arr)

    positions_arr += position

    return positions_arr


def initialize_detector_positions_reflected(
    parameters: AnyDetectorParameters,
    distance: float,
    reflection_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Initializes the 3D laboratory positions and 2D local detector coordinates
    for a rectangular or disk detector located at `distance` along the reflected beam axis
      k_re_hat = (sin(reflection_angle), 0, -cos(reflection_angle)),
    oriented perpendicular to k_re_hat.

    When `reflection_angle == 0.0`, this is identical to a detector at `z = -distance`
    parallel to the x-y plane.

    Returns:
        `(lab_positions_flat, local_coords_2d)`
        where `lab_positions_flat` has shape `(N1 * N2, 3)` and
        `local_coords_2d` has shape `(N1, N2, 3)` with `(x'_d, y'_d, -distance)`.
    """
    grid_x, grid_y = _create_detector_xy_grid(parameters)
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


def build_detector_grid(
    parameters: AnyDetectorParameters,
    distance: float,
    reflection_angle: float = 0.0,
    direction: int = -1,
) -> DetectorGrid:
    """Builds a complete `DetectorGrid` (lab positions, local coordinates, area weights,
    and outward surface normals) for either a rectangular or circular/disk detector.

    Args:
        parameters: `DetectorParameters` or `DiskDetectorParameters`.
        distance: Positive distance from the origin to the detector center.
        reflection_angle: Polar tilt angle (rad) in the x-z plane relative to `direction * z_hat`.
        direction: `-1` for backward/reflected placement along `(sin(r), 0, -cos(r))`
            or `+1` for forward placement along `(sin(r), 0, cos(r))`.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or 1, got {direction}")
    if distance <= 0.0:
        raise ValueError(f"distance must be positive, got {distance}")

    area_weights_2d = compute_detector_area_weights(parameters)

    if direction == -1:
        lab_positions_flat, local_coords_2d = initialize_detector_positions_reflected(
            parameters, distance=distance, reflection_angle=reflection_angle
        )
        normal_local = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        normal_lab = np.array(
            [np.sin(reflection_angle), 0.0, -np.cos(reflection_angle)], dtype=np.float64
        )
        return DetectorGrid(
            parameters=parameters,
            lab_positions_flat=lab_positions_flat,
            local_coords_2d=local_coords_2d,
            area_weights_2d=area_weights_2d,
            normal_lab=normal_lab,
            normal_local=normal_local,
        )

    grid_x, grid_y = _create_detector_xy_grid(parameters)
    dist = abs(distance)
    grid_z = np.full_like(grid_x, dist)
    local_coords_2d = np.stack((grid_x, grid_y, grid_z), axis=-1)
    normal_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if np.isclose(reflection_angle, 0.0):
        lab_positions_flat = local_coords_2d.reshape(-1, 3)
        normal_lab = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        cos_r = np.cos(reflection_angle)
        sin_r = np.sin(reflection_angle)
        lab_x = dist * sin_r + grid_x * cos_r
        lab_y = grid_y
        lab_z = dist * cos_r - grid_x * sin_r
        lab_positions_flat = np.stack((lab_x, lab_y, lab_z), axis=-1).reshape(-1, 3)
        normal_lab = np.array([sin_r, 0.0, cos_r], dtype=np.float64)

    return DetectorGrid(
        parameters=parameters,
        lab_positions_flat=lab_positions_flat,
        local_coords_2d=local_coords_2d,
        area_weights_2d=area_weights_2d,
        normal_lab=normal_lab,
        normal_local=normal_local,
    )


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
