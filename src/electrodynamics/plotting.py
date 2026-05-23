import numpy as np
from matplotlib.figure import Figure

from .typing import RealArray


def plot_particle_positions(fig: Figure, positions: RealArray) -> None:
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=3)  # pyright: ignore[reportArgumentType]

    ax = fig.add_subplot(1, 2, 2)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.scatter(positions[:, 0], positions[:, 1], s=3)
    ax.set_aspect(1.0)

    ax.grid()


def plot_angular_momentum_distribution(
    fig: Figure, initial_positions: RealArray, waist_radius: float, momenta: RealArray
) -> None:
    print("Plotting angular momentum distribution of initial conditions")

    ax = fig.add_subplot()

    ax.set_title("Normalized angular momentum distribution")

    momenta /= momenta.max()

    momenta = np.ma.masked_where(np.abs(momenta) < 1e-2, momenta)
    initial_positions = np.ma.masked_array(
        initial_positions,
        mask=np.repeat(np.ma.getmaskarray(momenta).reshape(-1, 1), 3, axis=1),
    )

    momenta = np.ma.compress_nd(momenta, axis=0)
    initial_positions = np.ma.compress_nd(initial_positions, axis=0)

    p = ax.scatter(
        initial_positions[:, 0] / waist_radius,
        initial_positions[:, 1] / waist_radius,
        c=momenta,
        cmap="bwr",
        s=3,
    )

    # ax.pcolormesh(Z, R, np.real(electric_field), vmin=-1, vmax=1.0, cmap="coolwarm")

    fig.colorbar(p)

    ax.set_xlabel("$x/w_0$")
    ax.set_ylabel("$y/w_0$")

    ax.set_aspect(1)

    ax.grid()

    fig.tight_layout()
