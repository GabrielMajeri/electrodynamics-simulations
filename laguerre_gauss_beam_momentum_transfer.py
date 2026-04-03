from cmath import polar
from dataclasses import dataclass
import wave

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]
type ComplexArray = npt.NDArray[np.complex128]


def generate_initial_particle_positions(R_max: float, num_points: int) -> Array:
    """Generates uniformly distributed points within a disk of radius R_max,
     centered in the origin.

    The disk lies in the x-y plane, determined by z = 0.
    """

    rng = np.random.default_rng(seed=17)

    angles = rng.uniform(low=0, high=2 * np.pi, size=num_points)
    radii_squared = rng.uniform(low=0, high=R_max**2, size=num_points)
    radii = np.sqrt(radii_squared)

    points = radii * np.vstack((np.cos(angles), np.sin(angles)))

    return np.vstack((points, np.zeros_like(angles))).T


def plot_particle_positions(fig: Figure, positions: Array) -> None:
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])  # pyright: ignore[reportArgumentType]


@dataclass
class PolarizationVector:
    x: complex
    y: complex

    def __init__(self, x: complex, y: complex):
        if not np.isclose(abs(x) ** 2 + abs(y) ** 2, 1):
            raise ValueError("|x|^2 + |y|^2 must be equal/close to 1")

        self.x = x
        self.y = y

    def to_numpy_array(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        return np.asarray((self.x, self.y), dtype=np.complex128)


def compute_gaussian_beam_electric_field(
    R: Array,
    Z: Array,
    amplitude: float,
    wavelength: float,
    polarization: PolarizationVector,
) -> ComplexArray:
    # w_0 = w(0)
    waist_radius = 2 * wavelength
    refractive_index = 1.0
    rayleigh_range = (np.pi * waist_radius**2 * refractive_index) / wavelength
    w_z = waist_radius * np.sqrt(1 + (Z / rayleigh_range) ** 2)

    # k
    wave_number = (2 * np.pi * refractive_index) / wavelength
    # R(z)
    curvature_radius = Z * (1 + (rayleigh_range / Z) ** 2)

    gouy_phase = np.arctan(Z / rayleigh_range)

    coefficients = (
        (waist_radius / w_z)
        * np.exp(-(R**2) / (w_z**2))
        * np.exp(
            -1j
            * (
                wave_number * Z
                + wave_number * (R**2) / (2 * curvature_radius)
                - gouy_phase
            )
        )
    )

    polarization_array = polarization.to_numpy_array()
    polarization_array = polarization_array.reshape(2, 1, 1)

    return amplitude * polarization_array * coefficients


def plot_electric_field_intensity(R: Array, Z: Array, electric_field: Array) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_title("Gaussian beam profile")

    # I've tried coloring each face/region based on the complex number's argument,
    # but it didn't look so good.
    # colors = plt.get_cmap("coolwarm")(np.angle(electric_field))

    ax.plot_surface(
        R,
        Z,
        np.abs(electric_field),
        cmap="coolwarm",
        # facecolors=colors,
        linewidth=0.5,
    )

    ax.set_box_aspect((2, 4, 3))

    ax.set_xlabel("$R$")
    ax.set_ylabel("$Z$")
    ax.set_zlabel("$|I|$")

    ax.grid()

    fig.tight_layout()
    fig.savefig("plots/gaussian_beam_intensity_profile.pdf")


def plot_electric_field_distribution(R: Array, Z: Array, electric_field: Array) -> None:
    fig = plt.figure(dpi=200)

    ax = fig.add_subplot()

    ax.set_title("Gaussian beam electric field distribution")

    ax.pcolormesh(Z, R, np.real(electric_field), vmin=-1, vmax=1.0, cmap="coolwarm")

    ax.set_xlabel("$z$")
    ax.set_ylabel("$r$")

    ax.grid()

    fig.tight_layout()
    fig.savefig("plots/gaussian_beam_electric_field_distribution.png")


def plot_electric_field_cross_section(
    rs: Array, zs: Array, electric_field: Array
) -> None:
    fig = plt.figure()

    ax = fig.add_subplot()

    ax.set_title("Gaussian beam cross-section at $z = 0$")

    z_0_index = len(zs) // 2

    ax.plot(rs, np.abs(electric_field[:, z_0_index]))

    ax.set_xlabel("$R$")
    ax.set_ylabel("$|I|$")

    ax.grid()

    fig.tight_layout()
    fig.savefig("plots/gaussian_beam_cross_section.pdf")


def compute_electric_and_magnetic_field_for_plane_wave(
    positions: Array, time: Array
) -> tuple[Array, Array]:
    assert positions.shape[-1] == 3, "Positions must be 3D vectors"

    z = positions[:, 2]

    E_0 = 1.0
    k = 1.0
    omega = 1.0

    E_x = E_0 * np.cos(k * z - omega * time)
    E_y = np.zeros_like(E_x)
    E_z = np.zeros_like(E_x)

    E = np.stack((E_x, E_y, E_z), axis=-1)

    B_0 = E_0

    B_y = B_0 * np.cos(k * z - omega * time)
    B_x = np.zeros_like(B_y)
    B_z = np.zeros_like(B_y)

    B = np.stack((B_x, B_y, B_z), axis=-1)

    # print("E =", E[0])
    # print("B =", B[0])

    return E, B


def compute_electric_and_magnetic_field_for_gaussian_beam(
    positions: Array, time: Array
) -> tuple[Array, Array]:
    assert positions.shape[-1] == 3, "Positions must be 3D vectors"

    x = positions[:, 0]
    y = positions[:, 1]
    r = np.sqrt(np.square(x) + np.square(y))
    z = positions[:, 2]

    # E_0
    amplitude = 1.0

    # Frequency / angular speed
    omega = 1.0

    # TODO: implement polarization support
    polarization = 1.0

    wavelength = 1

    # w_0 = w(0)
    waist_radius = 2 * wavelength
    refractive_index = 1.0
    rayleigh_range = (np.pi * waist_radius**2 * refractive_index) / wavelength
    w_z = waist_radius * np.sqrt(1 + (z / rayleigh_range) ** 2)

    # k
    wave_number = (2 * np.pi * refractive_index) / wavelength
    # R(z)
    zero_z = np.isclose(z, 0)
    curvature_radius = np.where(
        zero_z, np.inf, z * (1 + np.square(rayleigh_range / np.where(zero_z, 1, z)))
    )

    gouy_phase = np.arctan(z / rayleigh_range)

    coefficients = (
        (waist_radius / w_z)
        * np.exp(-(r**2) / (w_z**2))
        * np.exp(
            -1j
            * (
                wave_number * z
                + wave_number * (r**2) / (2 * curvature_radius)
                - gouy_phase
            )
            + 1j * omega * time
        )
    )

    # TODO: handle polarization
    E_x = amplitude * polarization * coefficients
    E_y = np.zeros_like(E_x)
    # TODO: better approximation formula for computing derivatives in the paraxial approximation
    E_z = (2j) / (wave_number * w_z**2) * (x * E_x + y * E_y)

    E = np.stack((E_x, E_y, E_z), axis=-1).real.astype(np.float64)

    c = 1.0
    B_x = -E_y / c
    B_y = E_x / c
    B_z = 1j / (omega * w_z**2) * (y * E_x - x * E_y)

    B = np.stack((B_x, B_y, B_z), axis=-1).real.astype(np.float64)

    return E, B


def compute_electromagnetic_field_tensor(positions: Array) -> Array:
    laboratory_time = positions[:, 0]
    position_vectors = positions[:, 1:4]

    E, B = compute_electric_and_magnetic_field_for_plane_wave(
        position_vectors, laboratory_time
    )

    # E, B = compute_electric_and_magnetic_field_for_gaussian_beam(
    #     position_vectors, laboratory_time
    # )

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


def simulate_trajectories(
    initial_positions: Array,
    start_time: float = 0.0,
    end_time: float = 50.0,
    time_step: float = 0.005,
) -> tuple[Array, Array, Array]:
    num_particles = len(initial_positions)
    duration = end_time - start_time
    num_timestamps = int(duration / time_step)

    positions = np.empty((num_timestamps, num_particles, 4))
    velocities = np.empty((num_timestamps, num_particles, 4))

    # Initialize trajectory data
    positions[0, :, 1:4] = initial_positions
    velocities[0, :] = 0
    velocities[0, :, 0] = 1

    # TODO
    particle_charge = 1
    particle_mass = 1

    # Each particle is treated as if it were in its own proper time frame
    proper_time = 0.0

    minkowski_metric = np.diag(np.array((1, -1, -1, -1), dtype=np.float64))

    # print("x0 =", initial_positions)

    for i in range(1, num_timestamps):
        field_tensor = compute_electromagnetic_field_tensor(positions[i - 1])

        v_lower_indices = velocities[i - 1] @ minkowski_metric

        # print("v_lower =", v_lower_indices)

        acceleration = (particle_charge / particle_mass) * (
            field_tensor @ v_lower_indices.reshape(num_particles, 4, 1)
        ).squeeze()

        # print("a =", acceleration)

        # (F u, u) should be 0
        assert np.isclose(0, np.dot(acceleration, v_lower_indices.squeeze()))

        # Symplectic Euler integration
        new_velocities = velocities[i - 1] + time_step * acceleration
        new_positions = positions[i - 1] + time_step * new_velocities

        positions[i] = new_positions
        velocities[i] = new_velocities

        # print("x_new =", new_positions)
        # print("v_new =", new_velocities)

        proper_time += time_step

    times = np.arange(start_time, end_time, time_step, dtype=np.float64)
    positions = np.permute_dims(positions, (1, 0, 2))
    velocities = np.permute_dims(velocities, (1, 0, 2))

    return times, positions, velocities


def main() -> int:
    # w_0
    # waist_radius = 75 * lmbd
    # R_max = 3.25 * waist_radius
    R_max = 1
    initial_positions = generate_initial_particle_positions(R_max, num_points=1)

    # Plot initial electron positions, for debugging purposes
    if False:
        fig = plt.figure()
        fig.suptitle("Initial electron positions")
        plot_particle_positions(fig, initial_positions)
        fig.savefig("initial_electron_positions.pdf")

    rs = np.linspace(-25, 25, 512)
    zs = np.linspace(-100, 100, 512)

    R, Z = np.meshgrid(rs, zs, sparse=False)

    # E_0
    amplitude = 1.0

    # Lambda
    wavelength = 3

    polarization = PolarizationVector(1.0, 0.0)

    electric_field = compute_gaussian_beam_electric_field(
        R, Z, amplitude, wavelength, polarization
    )

    electric_field = np.real(electric_field[0]).squeeze()

    if True:
        ### Electric field surface plot
        plot_electric_field_intensity(R, Z, electric_field)

    if True:
        ### Gaussian beam cross-sectional profile
        plot_electric_field_cross_section(rs, zs, electric_field)

    if True:
        ### Electric field distribution in the r-z plane
        plot_electric_field_distribution(R, Z, electric_field)

    times, positions, velocities = simulate_trajectories(initial_positions)

    # print(positions.shape)
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)

    ax.plot(times, positions[0, :, 0], label="$t$")
    ax.plot(times, positions[0, :, 1], label="$x$")
    ax.plot(times, positions[0, :, 2], label="$y$")
    ax.plot(times, positions[0, :, 3], label="$z$")

    ax.legend()

    ax = fig.add_subplot(1, 2, 2)

    ax.plot(times, velocities[0, :, 0], label="$c \\gamma$")
    ax.plot(times, velocities[0, :, 1], label="$v_x$")
    ax.plot(times, velocities[0, :, 2], label="$v_y$")
    ax.plot(times, velocities[0, :, 3], label="$v_z$")

    ax.legend()

    # ax = fig.add_subplot(projection="3d")

    # trajectory = positions[0]

    # ax.plot3D(trajectory[1], trajectory[2], trajectory[3])

    fig.savefig("plots/trajectories.pdf")

    return 0


if __name__ == "__main__":
    exit(main())
