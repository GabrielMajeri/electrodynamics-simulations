from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from electrodynamics.constants import lmbd

type Array = npt.NDArray[np.float64]


def generate_initial_particle_positions(R_max: float, num_points: int = 1000) -> Array:
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


def compute_gaussian_beam_electric_field(R: Array, Z: Array) -> Array:
    # E_0
    amplitude = 1.0

    # TODO
    polarization = 1.0

    wavelength = 1

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

    return amplitude * polarization * coefficients


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


def main() -> int:
    # w_0
    waist_radius = 75 * lmbd
    R_max = 3.25 * waist_radius
    initial_positions = generate_initial_particle_positions(R_max)

    # Plot initial electron positions, for debugging purposes
    if False:
        fig = plt.figure()
        fig.suptitle("Initial electron positions")
        plot_particle_positions(fig, initial_positions)
        fig.savefig("initial_electron_positions.pdf")

    rs = np.linspace(-25, 25, 512)
    zs = np.linspace(-100, 100, 512)

    R, Z = np.meshgrid(rs, zs, sparse=False)

    electric_field = compute_gaussian_beam_electric_field(R, Z)

    ### Electric field surface plot
    plot_electric_field_intensity(R, Z, electric_field)

    ### Gaussian beam cross-sectional profile
    plot_electric_field_cross_section(rs, zs, electric_field)

    return 0


if __name__ == "__main__":
    exit(main())
