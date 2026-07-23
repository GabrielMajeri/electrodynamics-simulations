import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        assert isinstance(self.axes, Axes3D), (
            "Arrow3D class should only be used with a 3D axes object"
        )
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def plot_particle_positions(fig: Figure, positions: np.ndarray) -> None:
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


def plot_final_momentum_distribution(
    fig: Figure,
    initial_positions: np.ndarray,
    waist_radius: float,
    momenta: np.ndarray,
    with_title: bool = True,
) -> None:
    print("Plotting z-momentum distribution for electrons")

    ax = fig.add_subplot()

    if with_title:
        ax.set_title("Normalized $z$ momentum distribution")

    momenta = momenta / np.abs(momenta).max()

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

    fig.colorbar(p)

    ax.set_xlabel("$x/w_0$")
    ax.set_ylabel("$y/w_0$")

    ax.set_aspect(1)

    ax.grid()

    fig.tight_layout()


def plot_angular_momentum_distribution(
    fig: Figure,
    initial_positions: np.ndarray,
    waist_radius: float,
    momenta: np.ndarray,
    with_title: bool = True,
) -> None:
    print("Plotting angular momentum distribution of initial conditions")

    ax = fig.add_subplot()

    if with_title:
        ax.set_title("Normalized angular momentum distribution")

    momenta = momenta / np.abs(momenta).max()

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
