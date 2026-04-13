import numpy as np

from .typing import RealArray


def generate_initial_positions_on_disk(
    generator: np.random.Generator, radius: float, num_points: int
) -> RealArray:
    """Generates uniformly distributed points within a disk of given radius,
    centered in the origin.

    The disk lies in the x-y plane, described by z = 0.
    """
    angles = generator.uniform(low=0, high=2 * np.pi, size=num_points)
    radii_squared = generator.uniform(low=0, high=radius**2, size=num_points)
    radii = np.sqrt(radii_squared).astype(np.float64)

    points = radii * np.vstack((np.cos(angles), np.sin(angles)))

    z = np.zeros_like(angles)
    return np.vstack((points, z)).T
