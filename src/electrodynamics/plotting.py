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

    ax.set_aspect(1)

    ax.grid()

    fig.tight_layout()
    fig.savefig("plots/angular_momentum_distribution.png")
