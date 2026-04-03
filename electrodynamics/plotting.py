from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from .typing import RealArray


def plot_particle_positions(fig: Figure, positions: RealArray) -> None:
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])  # pyright: ignore[reportArgumentType]


def plot_electric_field_intensity(
    R: RealArray, Z: RealArray, electric_field: RealArray
) -> None:
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


def plot_electric_field_distribution(
    R: RealArray, Z: RealArray, electric_field: RealArray
) -> None:
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
    rs: RealArray, zs: RealArray, electric_field: RealArray
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


def plot_angular_momentum_distribution(
    initial_positions: RealArray, momenta: RealArray
) -> None:
    fig = plt.figure(dpi=200)

    ax = fig.add_subplot()

    ax.set_title("Normalized angular momentum distribution")

    p = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], c=momenta)

    # ax.pcolormesh(Z, R, np.real(electric_field), vmin=-1, vmax=1.0, cmap="coolwarm")

    fig.colorbar(p)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.grid()

    fig.tight_layout()
    fig.savefig("plots/angular_momentum_distribution.png")
