import numpy as np

from .typing import RealArray


def generate_initial_positions_on_disk(
    generator: np.random.Generator, disk_radius: float, num_particles: int
) -> RealArray:
    """Generates uniformly distributed points within a disk of given radius,
    centered in the origin.

    The disk lies in the x-y plane, described by z = 0.
    """
    positions = np.empty((num_particles, 3), dtype=np.float64)

    for index in range(num_particles):
        radius = np.sqrt(generator.uniform(low=0, high=disk_radius**2))
        angle = generator.uniform(low=0, high=2 * np.pi)

        positions[index][0] = radius * np.cos(angle)
        positions[index][1] = radius * np.sin(angle)
        positions[index][2] = 0

    return positions
