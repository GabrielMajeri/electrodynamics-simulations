import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

from electrodynamics.detector import DetectorParameters


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


def plot_experimental_setup(
    fig: Figure,
    positions: np.ndarray,
    momenta: np.ndarray,
    detector_parameters: DetectorParameters,
    detector_positions: np.ndarray,
) -> None:
    assert len(positions) == len(momenta), (
        "Positions and momenta arrays should correspond to the same number of particles"
    )
    num_particles = len(positions)
    if num_particles > 1024:
        # Sample a subset of the initial particle bunch
        sample_indices = np.arange(1024)
        positions = positions[sample_indices]
        momenta = momenta[sample_indices]

    assert positions.shape[-1] in (3, 4), (
        "Positions should be an array of 3-vectors or 4-vectors"
    )
    if positions.shape[-1] == 4:
        positions = positions[:, 1:4]

    assert momenta.shape[-1] in (3, 4), (
        "Momenta should be an array of 3-vectors or 4-vectors"
    )
    if momenta.shape[-1] == 4:
        momenta = momenta[:, 1:4]

    fig.suptitle("Experimental setup")

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_title("Initial positions and momenta")

    # Draw the electrons
    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], s=3)  # pyright: ignore[reportArgumentType]

    # Draw initial momenta as arrows
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        momenta[:, 0],
        momenta[:, 1],
        momenta[:, 2],
        linewidth=1,
        length=5,
        normalize=True,
        color="orange",
    )

    # ax.set_zlim(-10, 10)

    ax = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    ax.set_title("Detector, laser beam and particles")

    # Draw electrons again
    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], s=3)  # pyright: ignore[reportArgumentType]

    # Draw the detector mesh
    detector_xs = detector_positions[:, 0].reshape(
        detector_parameters.grid_size_y, detector_parameters.grid_size_x
    )
    detector_ys = detector_positions[:, 1].reshape(
        detector_parameters.grid_size_y, detector_parameters.grid_size_x
    )
    detector_zs = detector_positions[:, 2].reshape(
        detector_parameters.grid_size_y, detector_parameters.grid_size_x
    )

    # Plot as a surface mesh
    ax.plot_surface(
        detector_xs,
        detector_ys,
        detector_zs,
        edgecolor="royalblue",
        lw=0.5,
        alpha=0.3,
        zorder=1,
    )

    # Plot laser direction as an arrow
    detector_z = detector_positions[0, 2]
    laser_beam_direction_arrow = Arrow3D(
        (0, 0),
        (0, 0),
        (detector_z * 0.75, detector_z * 0.25),
        color="red",
        lw=3,
        arrowstyle="-|>",
        mutation_scale=10,
        zorder=5,
    )
    ax.add_artist(laser_beam_direction_arrow)

    # fig.tight_layout()


def plot_final_momentum_distribution(
    fig: Figure,
    initial_positions: np.ndarray,
    waist_radius: float,
    momenta: np.ndarray,
    with_title: bool = True,
) -> None:
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


def plot_scattered_radiation_spectrum(
    fig: Figure,
    central_frequency: float,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
) -> None:
    axes = fig.subplots(2, 2)

    # Plot norm of complex vector and absolute values of components

    values = (
        np.asarray(np.linalg.vector_norm(spectrum, axis=-1)),
        np.asarray(np.abs(spectrum[:, 0])),
        np.asarray(np.abs(spectrum[:, 1])),
        np.asarray(np.abs(spectrum[:, 2])),
    )
    labels = ("|E(\\omega)|", "|E_x(\\omega)|", "|E_y(\\omega)|", "|E_z(\\omega)|")

    for index in range(4):
        row = index // 2
        column = index % 2

        ax: Axes = axes[row, column]

        ax.plot(frequencies, values[index])

        ax.axvline(central_frequency, linestyle="--", color="orange")

        ax.set_xlabel("Frequency $\\omega$")
        ax.set_ylabel("Amplitude")

        ax.set_title(f"${labels[index]}$")
        ax.grid()

    fig.tight_layout()
