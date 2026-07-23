import numpy as np

from electrodynamics.constants import SPEED_OF_LIGHT as c


def generate_initial_positions_uniformly_on_disk(
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


def generate_initial_positions_uniformly_within_ball(
    generator: np.random.Generator, ball_radius: float, num_particles: int
) -> np.ndarray:
    """Generates uniformly distributed points within the zero-centered ball of given radius."""
    # We work in 3D
    dimension = 3

    # Sample a set of vectors from multivariate standard normal distribution
    random_vectors = generator.normal(size=(num_particles, dimension))

    # Normalize them, turn them into unit vectors
    random_unit_vectors = random_vectors / np.expand_dims(
        np.linalg.vector_norm(random_vectors, axis=-1), axis=-1
    )

    # Scale them by some random radii, sampled proportionally
    # to the surface area of the corresponding sphere
    random_radii = generator.uniform(
        low=0.0, high=ball_radius, size=(num_particles, 1)
    ) ** (1 / dimension)

    return random_radii * random_unit_vectors


def generate_initial_particle_momenta_moving_towards_laser(
    num_particles: int, gamma: float, particle_mass: float
) -> np.ndarray:
    """Generates initial momenta for a bunch of particles
    moving relativistically towards the negative z axis.
    """

    initial_momenta = np.zeros(shape=(num_particles, 4), dtype=np.float64)

    initial_momenta[:, 0] = gamma * particle_mass * c
    initial_momenta[:, 3] = -np.sqrt(gamma**2 - 1) * particle_mass * c

    return initial_momenta


def generate_initial_particle_momenta_moving_in_direction(
    num_particles: int, gamma: float, particle_mass: float, direction: np.ndarray
) -> np.ndarray:
    """Generates initial momenta for a bunch of particles
    moving relativistically in a given direction.
    """

    initial_momenta = np.zeros(shape=(num_particles, 4), dtype=np.float64)

    normalized_direction = direction / np.linalg.norm(direction)

    initial_momenta[:, 0] = gamma * particle_mass * c
    initial_momenta[:, 1:] = (
        np.sqrt(gamma**2 - 1) * particle_mass * c * normalized_direction
    )

    return initial_momenta
