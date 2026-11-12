import numpy as np

from electrodynamics.constants import SPEED_OF_LIGHT as c


def generate_initial_positions_uniformly_on_disk(
    generator: np.random.Generator, disk_radius: float, num_particles: int
) -> np.ndarray:
    """Generates uniformly distributed points within a disk of given radius,
    centered in the origin.

    The disk lies in the x-y plane, described by z = 0.
    """
    positions = np.zeros((num_particles, 3), dtype=np.float64)

    radii = np.sqrt(generator.uniform(low=0.0, high=disk_radius**2, size=num_particles))
    angles = generator.uniform(low=0.0, high=2 * np.pi, size=num_particles)

    positions[:, 0] = radii * np.cos(angles)
    positions[:, 1] = radii * np.sin(angles)

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
    random_radii = ball_radius * generator.uniform(
        low=0.0, high=1, size=(num_particles, 1)
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


def generate_initial_positions_in_nanofiber(
    generator: np.random.Generator,
    fiber_radius: float,
    fiber_length: float,
    num_particles: int,
    z_start: float = 0.0,
) -> np.ndarray:
    """Generates uniformly distributed points within a cylindrical nanofiber
    of given radius and length, oriented along the z axis.
    The fiber spans from z = z_start to z = z_start + fiber_length.
    """
    positions = np.zeros((num_particles, 3), dtype=np.float64)

    radii = np.sqrt(
        generator.uniform(low=0.0, high=fiber_radius**2, size=num_particles)
    )
    angles = generator.uniform(low=0.0, high=2 * np.pi, size=num_particles)
    zs = generator.uniform(low=z_start, high=z_start + fiber_length, size=num_particles)

    positions[:, 0] = radii * np.cos(angles)
    positions[:, 1] = radii * np.sin(angles)
    positions[:, 2] = zs

    return positions


def generate_initial_positions_in_nanofiber_surface(
    generator: np.random.Generator,
    fiber_radius: float,
    skin_depth: float,
    fiber_length: float,
    num_particles: int,
    z_start: float = 0.0,
) -> np.ndarray:
    """Generates uniformly distributed points within the surface skin layer of a cylindrical nanofiber."""
    positions = np.zeros((num_particles, 3), dtype=np.float64)
    r_min = max(0.0, fiber_radius - skin_depth)

    radii = np.sqrt(
        generator.uniform(low=r_min**2, high=fiber_radius**2, size=num_particles)
    )
    angles = generator.uniform(low=0.0, high=2 * np.pi, size=num_particles)
    zs = generator.uniform(low=z_start, high=z_start + fiber_length, size=num_particles)

    positions[:, 0] = radii * np.cos(angles)
    positions[:, 1] = radii * np.sin(angles)
    positions[:, 2] = zs

    return positions


def generate_initial_particle_momenta_at_rest(
    num_particles: int, particle_mass: float
) -> np.ndarray:
    """Generates initial 4-momenta for particles at rest in the laboratory frame:
    u^mu = (m * c, 0, 0, 0).
    """
    initial_momenta = np.zeros(shape=(num_particles, 4), dtype=np.float64)
    initial_momenta[:, 0] = particle_mass * c
    return initial_momenta


def generate_initial_positions_on_thin_sheet(
    disk_radius: float,
    num_particles: int,
    z_center: float = 0.0,
    thickness: float = 0.0,
    sampling_method: str = "polar-grid",
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Generates initial 3D electron positions on a thin sheet (relativistic flying mirror)
    parallel to the x-y plane, centered at z = z_center with optional longitudinal thickness.

    Supported sampling methods:
    - 'polar-grid': Equal-area concentric rings (r_m, phi_n) with exact azimuthal symmetry,
      eliminating Monte Carlo shot noise in coherent far-field diffraction.
    - 'fibonacci': Deterministic golden-angle sunflower spiral on the disk.
    - 'random': Uniform random Monte Carlo sampling on the disk.
    """
    positions = np.zeros((num_particles, 3), dtype=np.float64)

    if sampling_method == "polar-grid":
        # Factor num_particles into (num_radial, num_azimuthal) with num_azimuthal >= 16 if possible
        target_azimuthal = max(8, round(np.sqrt(2.0 * num_particles)))
        best_n_phi = 1
        min_diff = num_particles
        for candidate in range(1, num_particles + 1):
            if num_particles % candidate == 0:
                diff = abs(candidate - target_azimuthal)
                if diff < min_diff:
                    min_diff = diff
                    best_n_phi = candidate
        n_phi = best_n_phi
        n_r = num_particles // n_phi

        r_indices = np.arange(n_r, dtype=np.float64)
        phi_indices = np.arange(n_phi, dtype=np.float64)

        radii_1d = disk_radius * np.sqrt((r_indices + 0.5) / n_r)
        angles_1d = (2.0 * np.pi * phi_indices) / n_phi

        r_grid, phi_grid = np.meshgrid(radii_1d, angles_1d, indexing="ij")
        radii = r_grid.reshape(-1)
        angles = phi_grid.reshape(-1)

        positions[:, 0] = radii * np.cos(angles)
        positions[:, 1] = radii * np.sin(angles)

    elif sampling_method == "fibonacci":
        indices = np.arange(num_particles, dtype=np.float64)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        radii = disk_radius * np.sqrt((indices + 0.5) / num_particles)
        angles = indices * golden_angle
        positions[:, 0] = radii * np.cos(angles)
        positions[:, 1] = radii * np.sin(angles)

    elif sampling_method == "random":
        if generator is None:
            generator = np.random.default_rng(42)
        radii = np.sqrt(
            generator.uniform(low=0.0, high=disk_radius**2, size=num_particles)
        )
        angles = generator.uniform(low=0.0, high=2.0 * np.pi, size=num_particles)
        positions[:, 0] = radii * np.cos(angles)
        positions[:, 1] = radii * np.sin(angles)

    else:
        raise ValueError(
            f"Unknown sampling_method '{sampling_method}'. Choose 'polar-grid', 'fibonacci', or 'random'."
        )

    if thickness > 0.0:
        if generator is None:
            generator = np.random.default_rng(42)
        positions[:, 2] = generator.uniform(
            low=z_center - thickness / 2.0,
            high=z_center + thickness / 2.0,
            size=num_particles,
        )
    else:
        positions[:, 2] = z_center

    return positions
