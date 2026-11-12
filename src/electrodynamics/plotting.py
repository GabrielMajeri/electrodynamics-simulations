import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch
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


def plot_microbunching_spatial_distribution(
    fig: Figure,
    positions: np.ndarray,
    laser_wavelength: float,
    fiber_radius: float,
    fiber_length: float,
    title: str = "Electron bunch spatial distribution",
) -> None:
    """Plots the spatial distribution of electrons showing periodic microbunching
    along the laser propagation axis (z) and radial confinement (r).
    """
    if positions.shape[-1] == 4:
        # Extract x, y, z
        x = positions[:, 1]
        y = positions[:, 2]
        z = positions[:, 3]
    else:
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

    r = np.hypot(x, y)
    z_norm = z / laser_wavelength
    r_norm = r / laser_wavelength

    fig.set_size_inches(10, 4.5)
    axes = fig.subplots(1, 2)
    fig.suptitle(title)

    # Subplot 1: (z, r) scatter with nanofiber region
    ax0: Axes = axes[0]
    ax0.set_title("Electron positions $(z, r)$")
    # Draw nanofiber bounds
    fiber_z = [
        0,
        fiber_length / laser_wavelength,
        fiber_length / laser_wavelength,
        0,
        0,
    ]
    fiber_r = [
        0,
        0,
        fiber_radius / laser_wavelength,
        fiber_radius / laser_wavelength,
        0,
    ]
    ax0.plot(fiber_z, fiber_r, color="green", linestyle="--", label="Nanofiber outline")

    ax0.scatter(z_norm, r_norm, s=4, alpha=0.6, color="red", label="Electrons")
    ax0.set_xlabel("$z / \\lambda_0$")
    ax0.set_ylabel("$r / \\lambda_0$")
    ax0.grid(True)
    ax0.legend(loc="upper right")

    # Subplot 2: Longitudinal density histogram dN/dz
    ax1: Axes = axes[1]
    ax1.set_title("Longitudinal density $dN/dz$")
    z_min, z_max = z_norm.min(), z_norm.max()
    num_bins = max(30, int((z_max - z_min) * 15)) if z_max > z_min else 50
    ax1.hist(z_norm, bins=num_bins, color="royalblue", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("$z / \\lambda_0$")
    ax1.set_ylabel("Electron count")
    ax1.grid(True)

    fig.tight_layout()


def plot_bunch_phase_space(
    fig: Figure,
    positions: np.ndarray,
    momenta: np.ndarray,
    laser_wavelength: float,
    title: str = "Longitudinal phase space $(z, p_z)$",
) -> None:
    """Plots longitudinal phase-space (z, p_z) showing electron phase-locking."""
    z = positions[:, 3] if positions.shape[-1] == 4 else positions[:, 2]
    pz = momenta[:, 3] if momenta.shape[-1] == 4 else momenta[:, 2]

    ax = fig.add_subplot()
    fig.suptitle(title)

    scatter = ax.scatter(
        z / laser_wavelength,
        pz / 137.036,  # Normalized by m_e * c (since m_e = 1, c = 137.036)
        s=4,
        c=momenta[:, 0] / 137.036,  # Color by relativistic gamma
        cmap="plasma",
        alpha=0.7,
    )
    fig.colorbar(scatter, label="Relativistic factor $\\gamma$")

    ax.set_xlabel("$z / \\lambda_0$")
    ax.set_ylabel("$p_z / (m_e c)$")
    ax.grid(True)
    fig.tight_layout()


def plot_transverse_bunch_profile(
    fig: Figure,
    positions: np.ndarray,
    laser_wavelength: float,
    waist_radius: float,
    title: str = "Transverse electron distribution",
) -> None:
    """Plots transverse distribution (x, y) showing doughnut / disk profile."""
    x = positions[:, 1] if positions.shape[-1] == 4 else positions[:, 0]
    y = positions[:, 2] if positions.shape[-1] == 4 else positions[:, 1]

    ax = fig.add_subplot()
    fig.suptitle(title)

    r_norm = np.hypot(x, y) / laser_wavelength
    scatter = ax.scatter(
        x / laser_wavelength,
        y / laser_wavelength,
        s=5,
        c=r_norm,
        cmap="coolwarm",
        alpha=0.7,
    )
    fig.colorbar(scatter, label="$r / \\lambda_0$")

    # Draw waist circle
    waist_circle = Circle(
        (0, 0),
        waist_radius / laser_wavelength,
        color="red",
        fill=False,
        linestyle="--",
        label="$w_0$",
    )
    ax.add_patch(waist_circle)

    ax.set_xlabel("$x / \\lambda_0$")
    ax.set_ylabel("$y / \\lambda_0$")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


def plot_energy_spectrum(
    fig: Figure,
    momenta: np.ndarray,
    title: str = "Electron kinetic energy spectrum",
) -> None:
    """Plots kinetic energy spectrum in MeV."""
    # In atomic units, rest mass m_e * c^2 = 137.036^2 a.u. = 0.5109989 MeV
    gamma = momenta[:, 0] / 137.036
    kinetic_energy_mev = (gamma - 1.0) * 0.5109989

    ax = fig.add_subplot()
    fig.suptitle(title)

    num_bins = min(80, max(20, len(momenta) // 20))
    ax.hist(
        kinetic_energy_mev,
        bins=num_bins,
        color="crimson",
        edgecolor="black",
        alpha=0.75,
    )
    ax.set_xlabel("Kinetic energy $\\mathcal{E}_k$ [MeV]")
    ax.set_ylabel("Electron count $dN / d\\mathcal{E}_k$")
    ax.grid(True)
    fig.tight_layout()


def plot_transverse_velocity_vortex(
    fig: Figure,
    positions: np.ndarray,
    momenta: np.ndarray,
    laser_wavelength: float,
    title: str = "Transverse velocity field $(v_x, v_y)$",
) -> None:
    """Plots transverse velocity field showing beam angular momentum (vortex rotation)."""
    x = positions[:, 1] if positions.shape[-1] == 4 else positions[:, 0]
    y = positions[:, 2] if positions.shape[-1] == 4 else positions[:, 1]
    px = momenta[:, 1]
    py = momenta[:, 2]
    gamma = momenta[:, 0] / 137.036

    vx = (px / gamma) / 137.036  # v_x / c
    vy = (py / gamma) / 137.036  # v_y / c

    ax = fig.add_subplot()
    fig.suptitle(title)

    # Subsample if many particles
    n = len(x)
    if n > 512:
        idx = np.random.default_rng(42).choice(n, size=512, replace=False)
        x_sub, y_sub, vx_sub, vy_sub = x[idx], y[idx], vx[idx], vy[idx]
    else:
        x_sub, y_sub, vx_sub, vy_sub = x, y, vx, vy

    # Azimuthal momentum / angular momentum per particle
    L_z = x * py - y * px

    scatter = ax.scatter(
        x / laser_wavelength,
        y / laser_wavelength,
        c=L_z,
        cmap="bwr",
        s=6,
        alpha=0.5,
    )
    fig.colorbar(scatter, label="$L_z$ [a.u.]")

    ax.quiver(
        x_sub / laser_wavelength,
        y_sub / laser_wavelength,
        vx_sub,
        vy_sub,
        color="black",
        scale=2.0,
        width=0.003,
        alpha=0.8,
    )

    ax.set_xlabel("$x / \\lambda_0$")
    ax.set_ylabel("$y / \\lambda_0$")
    ax.set_aspect("equal")
    ax.grid(True)
    fig.tight_layout()


def plot_beam_angular_momentum_evolution(
    fig: Figure,
    timestamps: np.ndarray,
    total_bam: np.ndarray,
    laser_period: float,
    title: str = "Beam angular momentum evolution",
) -> None:
    """Plots evolution of total BAM L_z vs time (in laser periods)."""
    ax = fig.add_subplot()
    fig.suptitle(title)

    ax.plot(timestamps / laser_period, total_bam, color="purple", lw=2)
    ax.set_xlabel("Time $t / T_0$")
    ax.set_ylabel("Total BAM $\\mathcal{L}_z$ [a.u.]")
    ax.grid(True)
    fig.tight_layout()


def plot_detector_radiation_and_oam(
    fig: Figure,
    detector_positions: np.ndarray,
    electric_field: np.ndarray,
    dL_z: np.ndarray,
    laser_wavelength: float,
    title: str = "Scattered radiation and radiated OAM on detector",
) -> None:
    """Plots 2D detector patterns:
    1. Radiation intensity |E|^2
    2. Optical vortex phase arg(E_x + i E_y)
    3. Radiated orbital angular momentum density Re(dL_z)

    detector_positions: shape (Nx, Ny, 3)
    electric_field: shape (Nx, Ny, 3)
    dL_z: shape (Nx, Ny)
    """
    fig.set_size_inches(15, 4.5)
    axes = fig.subplots(1, 3)
    fig.suptitle(title, fontsize=12)

    # Coordinates in units of laser wavelength
    xs = detector_positions[:, 0, 0] / laser_wavelength
    ys = detector_positions[0, :, 1] / laser_wavelength
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]

    # 1. Radiation intensity: |E|^2 = |E_x|^2 + |E_y|^2 + |E_z|^2
    intensity = np.sum(np.abs(electric_field) ** 2, axis=-1)
    im0 = axes[0].imshow(intensity.T, origin="lower", extent=extent, cmap="inferno")
    axes[0].set_title(r"Radiation Intensity $|E_{\mathrm{det}}|^2$")
    axes[0].set_xlabel(r"$x / \lambda_0$")
    axes[0].set_ylabel(r"$y / \lambda_0$")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Transverse phase arg(E_x + i E_y) (helical vortex wavefront)
    E_trans = electric_field[:, :, 0] + 1j * electric_field[:, :, 1]
    phase = np.angle(E_trans)
    im1 = axes[1].imshow(
        phase.T, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    axes[1].set_title(r"Vortex Phase $\arg(E_x + i E_y)$")
    axes[1].set_xlabel(r"$x / \lambda_0$")
    axes[1].set_ylabel(r"$y / \lambda_0$")
    axes[1].set_aspect("equal")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar1.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    # 3. Radiated OAM density Re(dL_z)
    dL_z_real = np.real(dL_z)
    total_Lz = float(np.sum(dL_z_real))
    vmax = max(float(np.max(np.abs(dL_z_real))), 1e-12)
    im2 = axes[2].imshow(
        dL_z_real.T, origin="lower", extent=extent, cmap="bwr", vmin=-vmax, vmax=vmax
    )
    axes[2].set_title(
        f"Radiated OAM Density $\\mathrm{{Re}}(dL_z)$\n(Total $L_z^{{\\mathrm{{EM}}}} = {total_Lz:.3e}$)"
    )
    axes[2].set_xlabel(r"$x / \lambda_0$")
    axes[2].set_ylabel(r"$y / \lambda_0$")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
