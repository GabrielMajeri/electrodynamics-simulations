import numpy as np

from electrodynamics.constants import SPEED_OF_LIGHT


def generate_initial_positions_on_disk(
    generator: np.random.Generator, disk_radius: float, num_particles: int
) -> np.ndarray:
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


def generate_initial_particle_momenta_moving_towards_laser(
    num_particles: int, gamma: float, particle_mass: float
) -> np.ndarray:
    """Generates initial momenta for a bunch of particles
    moving relativistically towards the negative z axis.
    """

    initial_momenta = np.zeros(shape=(num_particles, 4), dtype=np.float64)

    c = SPEED_OF_LIGHT

    initial_momenta[:, 0] = gamma * particle_mass * c
    initial_momenta[:, 3] = -np.sqrt(gamma**2 - 1) * particle_mass * c

    return initial_momenta
