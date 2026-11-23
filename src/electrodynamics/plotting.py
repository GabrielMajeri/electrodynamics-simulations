from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

from electrodynamics.angular_momentum import (
    compute_particle_angular_momentum_z,
    integrate_detector_oam_spectral_density,
)
from electrodynamics.constants import ELECTRON_REST_ENERGY_MEV, SPEED_OF_LIGHT
from electrodynamics.detector import (
    AnyDetectorParameters,
    DiskDetectorParameters,
    compute_detector_area_weights,
    compute_disk_radial_quadrature,
)

if TYPE_CHECKING:
    from electrodynamics.integrate import SnapshotData


def save_figure(
    fig: Figure,
    stem: Path,
    formats: Sequence[str] = ("pdf",),
    close: bool = True,
) -> None:
    "Saves `fig` to `{stem}.{ext}` for each extension in `formats` and optionally closes it."
    for ext in formats:
        fig.savefig(stem.with_suffix(f".{ext}"))
    if close:
        plt.close(fig)


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
    detector_parameters: AnyDetectorParameters,
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

    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], s=3)  # pyright: ignore[reportArgumentType]

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

    ax = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    ax.set_title("Detector, laser beam and particles")

    ax.scatter3D(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],  # pyright: ignore[reportArgumentType]
        s=3,
        rasterized=True,
    )

    detector_grid = detector_parameters.reshape_to_grid(detector_positions)
    if isinstance(detector_parameters, DiskDetectorParameters):
        detector_grid = np.concatenate([detector_grid, detector_grid[:, :1, :]], axis=1)

    ax.plot_surface(
        detector_grid[:, :, 0],
        detector_grid[:, :, 1],
        detector_grid[:, :, 2],
        edgecolor="royalblue",
        lw=0.5,
        alpha=0.3,
        zorder=1,
        rasterized=True,
    )

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


def _plot_normalized_transverse_distribution(
    fig: Figure,
    initial_positions: np.ndarray,
    waist_radius: float,
    values: np.ndarray,
    title: str,
    with_title: bool = True,
) -> None:
    ax = fig.add_subplot()

    if with_title:
        ax.set_title(title)

    max_abs = np.abs(values).max()
    if max_abs > 0:
        values = values / max_abs

    masked_values = np.ma.masked_where(np.abs(values) < 1e-2, values)
    masked_positions = np.ma.masked_array(
        initial_positions,
        mask=np.repeat(np.ma.getmaskarray(masked_values).reshape(-1, 1), 3, axis=1),
    )

    compressed_values = np.ma.compress_nd(masked_values, axis=0)
    compressed_positions = np.ma.compress_nd(masked_positions, axis=0)

    p = ax.scatter(
        compressed_positions[:, 0] / waist_radius,
        compressed_positions[:, 1] / waist_radius,
        c=compressed_values,
        cmap="bwr",
        s=3,
        rasterized=True,
    )

    fig.colorbar(p)

    ax.set_xlabel("$x/w_0$")
    ax.set_ylabel("$y/w_0$")
    ax.set_aspect(1)
    ax.grid()
    fig.tight_layout()


def plot_final_momentum_distribution(
    fig: Figure,
    initial_positions: np.ndarray,
    waist_radius: float,
    momenta: np.ndarray,
    with_title: bool = True,
) -> None:
    _plot_normalized_transverse_distribution(
        fig,
        initial_positions,
        waist_radius,
        momenta,
        title="Normalized $z$ momentum distribution",
        with_title=with_title,
    )


def plot_angular_momentum_distribution(
    fig: Figure,
    initial_positions: np.ndarray,
    waist_radius: float,
    momenta: np.ndarray,
    with_title: bool = True,
) -> None:
    _plot_normalized_transverse_distribution(
        fig,
        initial_positions,
        waist_radius,
        momenta,
        title="Normalized angular momentum distribution",
        with_title=with_title,
    )


def plot_scattered_radiation_spectrum(
    fig: Figure,
    central_frequency: float,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
) -> None:
    axes = fig.subplots(2, 2)

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
        x, y, z = positions[:, 1], positions[:, 2], positions[:, 3]
    else:
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    r = np.hypot(x, y)
    z_norm = z / laser_wavelength
    r_norm = r / laser_wavelength

    fig.set_size_inches(10, 4.5)
    axes = fig.subplots(1, 2)
    fig.suptitle(title)

    ax0: Axes = axes[0]
    ax0.set_title("Electron positions $(z, r)$")
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
        pz / SPEED_OF_LIGHT,
        s=4,
        c=momenta[:, 0] / SPEED_OF_LIGHT,
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
    gamma = momenta[:, 0] / SPEED_OF_LIGHT
    kinetic_energy_mev = (gamma - 1.0) * ELECTRON_REST_ENERGY_MEV

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
    gamma = momenta[:, 0] / SPEED_OF_LIGHT

    vx = (px / gamma) / SPEED_OF_LIGHT
    vy = (py / gamma) / SPEED_OF_LIGHT

    ax = fig.add_subplot()
    fig.suptitle(title)

    n = len(x)
    if n > 512:
        idx = np.random.default_rng(42).choice(n, size=512, replace=False)
        x_sub, y_sub, vx_sub, vy_sub = x[idx], y[idx], vx[idx], vy[idx]
    else:
        x_sub, y_sub, vx_sub, vy_sub = x, y, vx, vy

    L_z = compute_particle_angular_momentum_z(positions, momenta)

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


def _infer_is_polar_grid(
    detector_positions: np.ndarray,
    detector_parameters: AnyDetectorParameters | None = None,
) -> bool:
    if detector_parameters is not None:
        return isinstance(detector_parameters, DiskDetectorParameters)
    x = detector_positions[:, :, 0]
    y = detector_positions[:, :, 1]
    r = np.hypot(x, y)
    return bool(not np.allclose(x[:, :1], x) and np.allclose(r[:, :1], r))


def _build_polar_pcolormesh_edges(
    detector_positions: np.ndarray, disk_radius: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    radii_1d = np.hypot(detector_positions[:, 0, 0], detector_positions[:, 0, 1])
    n_r = len(radii_1d)
    n_phi = detector_positions.shape[1]

    if disk_radius is not None:
        r_max = float(disk_radius)
    elif n_r > 1:
        r_max = float(radii_1d[-1] + 0.5 * (radii_1d[-1] - radii_1d[-2]))
    else:
        r_max = float(2.0 * radii_1d[0])
    r_max = max(r_max, float(radii_1d[-1]))

    r_edges = np.empty(n_r + 1, dtype=np.float64)
    r_edges[0] = 0.0
    if n_r > 1:
        r_edges[1:-1] = 0.5 * (radii_1d[:-1] + radii_1d[1:])
    r_edges[-1] = r_max

    angles_1d = np.arctan2(detector_positions[0, :, 1], detector_positions[0, :, 0])
    dphi = (2.0 * np.pi) / n_phi
    phi_edges = (
        angles_1d[0] - 0.5 * dphi + dphi * np.arange(n_phi + 1, dtype=np.float64)
    )

    X_edge = r_edges[:, None] * np.cos(phi_edges[None, :])
    Y_edge = r_edges[:, None] * np.sin(phi_edges[None, :])
    return X_edge, Y_edge, r_max


def _infer_polar_area_weights(
    detector_positions: np.ndarray,
    detector_parameters: AnyDetectorParameters | None = None,
) -> np.ndarray:
    if isinstance(detector_parameters, DiskDetectorParameters):
        return compute_detector_area_weights(detector_parameters)
    X_edge, Y_edge, _ = _build_polar_pcolormesh_edges(detector_positions)
    r_edges = np.hypot(X_edge[:, 0], Y_edge[:, 0])
    n_phi = detector_positions.shape[1]
    dphi = (2.0 * np.pi) / n_phi
    radial_annuli = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    return np.broadcast_to(
        radial_annuli[:, None] * dphi, detector_positions.shape[:2]
    ).copy()


def _plot_polar_panel(
    ax: Axes,
    X_edge: np.ndarray,
    Y_edge: np.ndarray,
    values: np.ndarray,
    r_max: float,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    mesh = ax.pcolormesh(
        X_edge, Y_edge, values, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0), r_max, fill=False, edgecolor="black", linewidth=0.8, alpha=0.6
        )
    )
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    return mesh


def infer_detector_area_weights(
    detector_positions: np.ndarray,
    detector_parameters: AnyDetectorParameters | None = None,
) -> np.ndarray | None:
    """Infers area quadrature weights when `detector_parameters` or polar geometry is present."""
    if detector_parameters is not None:
        return compute_detector_area_weights(detector_parameters)
    if _infer_is_polar_grid(detector_positions):
        return _infer_polar_area_weights(detector_positions)
    return None


def plot_detector_scalar_panel(
    ax: Axes,
    detector_positions: np.ndarray,
    values_2d: np.ndarray,
    scale: float,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    detector_parameters: AnyDetectorParameters | None = None,
):
    """Renders a 2D detector scalar map onto `ax`, automatically handling rectangular
    (`imshow`) and circular disk (`pcolormesh`) geometries scaled by `scale`.
    """
    if _infer_is_polar_grid(detector_positions, detector_parameters):
        disk_radius = (
            detector_parameters.radius
            if isinstance(detector_parameters, DiskDetectorParameters)
            else None
        )
        X_edge, Y_edge, r_max = _build_polar_pcolormesh_edges(
            detector_positions, disk_radius=disk_radius
        )
        return _plot_polar_panel(
            ax,
            X_edge / scale,
            Y_edge / scale,
            values_2d,
            r_max / scale,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    xs = detector_positions[:, 0, 0] / scale
    ys = detector_positions[0, :, 1] / scale
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    image = ax.imshow(
        values_2d.T, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_aspect("equal")
    return image


def plot_detector_radiation_and_oam(
    fig: Figure,
    detector_positions: np.ndarray,
    electric_field: np.ndarray,
    dL_z: np.ndarray,
    laser_wavelength: float,
    title: str = "Scattered radiation and radiated OAM on detector",
    wave_number: float | None = None,
    area_weights: np.ndarray | None = None,
    detector_parameters: AnyDetectorParameters | None = None,
) -> None:
    """Plots 2D detector patterns:
    1. Radiation intensity |E|^2
    2. Optical vortex phase arg(E_x) (optionally compensated for spherical wavefront curvature)
    3. Radiated orbital angular momentum spectral density Re(d^2 L_z / (d omega dA))

    Supports both rectangular `(Nx, Ny, ...)` and polar disk `(Nr, Nphi, ...)` grids.
    """
    fig.set_size_inches(15, 4.5)
    axes = fig.subplots(1, 3)
    fig.suptitle(title, fontsize=12)

    is_polar = _infer_is_polar_grid(detector_positions, detector_parameters)

    if (
        np.max(np.abs(electric_field[:, :, 0]))
        >= np.max(np.abs(electric_field[:, :, 1])) * 0.5
    ):
        E_comp = electric_field[:, :, 0]
    else:
        E_comp = electric_field[:, :, 1]

    if wave_number is not None:
        r_detector = np.linalg.norm(detector_positions, axis=-1)
        if is_polar:
            r_center = abs(float(detector_positions[0, 0, 2]))
        else:
            r_center = abs(
                float(
                    detector_positions[
                        detector_positions.shape[0] // 2,
                        detector_positions.shape[1] // 2,
                        2,
                    ]
                )
            )
        E_comp = E_comp * np.exp(-1j * wave_number * (r_detector - r_center))

    intensity = np.sum(np.abs(electric_field) ** 2, axis=-1)
    phase = np.angle(E_comp)
    dL_z_real = np.real(dL_z)
    vmax = max(float(np.max(np.abs(dL_z_real))), 1e-30)

    eff_weights = (
        area_weights
        if area_weights is not None
        else infer_detector_area_weights(detector_positions, detector_parameters)
    )
    total_Lz = integrate_detector_oam_spectral_density(
        detector_positions, dL_z, area_weights=eff_weights
    )

    im0 = plot_detector_scalar_panel(
        axes[0],
        detector_positions,
        intensity,
        laser_wavelength,
        cmap="inferno",
        detector_parameters=detector_parameters,
    )
    im1 = plot_detector_scalar_panel(
        axes[1],
        detector_positions,
        phase,
        laser_wavelength,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        detector_parameters=detector_parameters,
    )
    im2 = plot_detector_scalar_panel(
        axes[2],
        detector_positions,
        dL_z_real,
        laser_wavelength,
        cmap="bwr",
        vmin=-vmax,
        vmax=vmax,
        detector_parameters=detector_parameters,
    )

    axes[0].set_title(r"Radiation Intensity $|E_{\mathrm{det}}|^2$")
    axes[0].set_xlabel(r"$x / \lambda_0$")
    axes[0].set_ylabel(r"$y / \lambda_0$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].set_title(r"Vortex Phase $\arg(E_{\perp})$")
    axes[1].set_xlabel(r"$x / \lambda_0$")
    axes[1].set_ylabel(r"$y / \lambda_0$")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar1.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    axes[2].set_title(
        f"Radiated OAM Spectral Density $\\mathrm{{Re}}(d^2 L_z / d\\omega\\,dA)$\n($dL_z^{{\\mathrm{{EM}}}}/d\\omega = {total_Lz:.3e}$)"
    )
    axes[2].set_xlabel(r"$x / \lambda_0$")
    axes[2].set_ylabel(r"$y / \lambda_0$")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()


def plot_detector_field_map(
    fig: Figure,
    values_2d: np.ndarray,
    extent: tuple[float, float, float, float],
    title: str,
    suptitle: str = "Scattered electric field on detector",
    cmap: str = "bwr",
    symmetric_limits: bool = True,
    local_coords_2d: np.ndarray | None = None,
    scale: float = 1.0,
    disk_radius: float | None = None,
) -> None:
    "Plots a 2D scalar slice `(Nx, Ny)` or `(Nr, Nphi)` on the detector screen."
    ax = fig.add_subplot()
    fig.suptitle(suptitle)
    ax.set_title(title)

    if local_coords_2d is not None and _infer_is_polar_grid(local_coords_2d):
        X_edge, Y_edge, r_max = _build_polar_pcolormesh_edges(
            local_coords_2d, disk_radius=disk_radius
        )
        X_edge_scaled = X_edge / scale
        Y_edge_scaled = Y_edge / scale
        r_max_scaled = r_max / scale
        if symmetric_limits:
            vmax = max(float(np.max(np.abs(values_2d))), 1e-15)
            image = _plot_polar_panel(
                ax,
                X_edge_scaled,
                Y_edge_scaled,
                values_2d,
                r_max_scaled,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            image = _plot_polar_panel(
                ax, X_edge_scaled, Y_edge_scaled, values_2d, r_max_scaled, cmap=cmap
            )
    else:
        data = values_2d.T
        if symmetric_limits:
            vmax = max(float(np.max(np.abs(data))), 1e-15)
            image = ax.imshow(
                data, cmap=cmap, origin="lower", extent=extent, vmin=-vmax, vmax=vmax
            )
        else:
            image = ax.imshow(data, cmap=cmap, origin="lower", extent=extent)

    ax.set_xlabel("Detector $x/w_0$")
    ax.set_ylabel("Detector $y/w_0$")
    ax.grid()
    fig.colorbar(image)
    fig.tight_layout()


def save_detector_field_component_plots(
    output_directory: Path,
    electric_field_2d: np.ndarray,
    detector_parameters: AnyDetectorParameters,
    waist_radius: float,
    formats: Sequence[str] = ("pdf", "png"),
    local_coords_2d: np.ndarray | None = None,
) -> None:
    "Saves individual `E_x`, `E_y`, `E_z`, and `|E|` detector field maps."
    if isinstance(detector_parameters, DiskDetectorParameters):
        r_norm = detector_parameters.radius / waist_radius
        detector_extent: tuple[float, float, float, float] = (
            -r_norm,
            +r_norm,
            -r_norm,
            +r_norm,
        )
        if local_coords_2d is None:
            radii_1d, _, angles_1d = compute_disk_radial_quadrature(detector_parameters)
            r_grid, phi_grid = np.meshgrid(radii_1d, angles_1d, indexing="ij")
            coords_for_plot = np.stack(
                (
                    r_grid * np.cos(phi_grid),
                    r_grid * np.sin(phi_grid),
                    np.zeros_like(r_grid),
                ),
                axis=-1,
            )
        else:
            coords_for_plot = local_coords_2d
        disk_radius: float | None = detector_parameters.radius
    else:
        detector_extent = (
            (-detector_parameters.width / 2) / waist_radius,
            (+detector_parameters.width / 2) / waist_radius,
            (-detector_parameters.height / 2) / waist_radius,
            (+detector_parameters.height / 2) / waist_radius,
        )
        coords_for_plot = None
        disk_radius = None

    for index, coordinate in enumerate(("x", "y", "z")):
        fig = plt.figure(figsize=(7, 6), dpi=200)
        plot_detector_field_map(
            fig,
            np.real(electric_field_2d[:, :, index]),
            detector_extent,
            title=f"$E_{coordinate}$",
            cmap="bwr",
            symmetric_limits=True,
            local_coords_2d=coords_for_plot,
            scale=waist_radius,
            disk_radius=disk_radius,
        )
        save_figure(fig, output_directory / f"electric_field_{coordinate}", formats)

    fig = plt.figure(figsize=(7, 6), dpi=200)
    plot_detector_field_map(
        fig,
        np.linalg.vector_norm(electric_field_2d, axis=-1),
        detector_extent,
        title="$|E|$",
        cmap="inferno",
        symmetric_limits=False,
        local_coords_2d=coords_for_plot,
        scale=waist_radius,
        disk_radius=disk_radius,
    )
    save_figure(fig, output_directory / "electric_field_norm", formats)


def save_nanofiber_snapshot_plots(
    output_directory: Path,
    snapshots: Sequence[SnapshotData],
    laser_wavelength: float,
    fiber_radius: float,
    fiber_length: float,
    waist_radius: float,
    j_z: int,
) -> None:
    """Generates and saves spatial microbunching, longitudinal phase-space,
    and transverse profile plots for each recorded nanofiber snapshot.
    """
    for snap in snapshots:
        t_label = f"{snap.time:.1f}T0"
        pos_np = np.asarray(snap.positions)
        mom_np = np.asarray(snap.momenta)

        fig = plt.figure(dpi=200)
        plot_microbunching_spatial_distribution(
            fig,
            pos_np,
            laser_wavelength,
            fiber_radius,
            fiber_length,
            title=f"Electron bunches at $t = {snap.time:.0f} T_0$ ($j_z = {j_z}$)",
        )
        save_figure(fig, output_directory / f"microbunching_spatial_{t_label}")

        fig = plt.figure(dpi=200)
        plot_bunch_phase_space(
            fig,
            pos_np,
            mom_np,
            laser_wavelength,
            title=f"Longitudinal phase space $(z, p_z)$ at $t = {snap.time:.0f} T_0$",
        )
        save_figure(fig, output_directory / f"phase_space_{t_label}")

        fig = plt.figure(dpi=200)
        plot_transverse_bunch_profile(
            fig,
            pos_np,
            laser_wavelength,
            waist_radius,
            title=f"Transverse distribution $(x, y)$ at $t = {snap.time:.0f} T_0$ ($j_z = {j_z}$)",
        )
        save_figure(fig, output_directory / f"transverse_profile_{t_label}")


def save_nanofiber_final_plots(
    output_directory: Path,
    final_positions: np.ndarray,
    final_momenta: np.ndarray,
    all_lz_history: Sequence[np.ndarray],
    laser_wavelength: float,
    laser_period: float,
    t_final_periods: float,
    j_z: int,
) -> None:
    """Generates and saves final energy spectrum, transverse velocity vortex,
    and beam angular momentum evolution plots for nanofiber simulations.
    """
    fig = plt.figure(dpi=200)
    plot_energy_spectrum(
        fig,
        final_momenta,
        title=f"Electron kinetic energy spectrum at $t = {t_final_periods:.0f} T_0$",
    )
    save_figure(fig, output_directory / "final_energy_spectrum")

    fig = plt.figure(dpi=200)
    plot_transverse_velocity_vortex(
        fig,
        final_positions,
        final_momenta,
        laser_wavelength,
        title=f"Transverse velocity field $(v_x, v_y)$ at $t = {t_final_periods:.0f} T_0$ ($j_z = {j_z}$)",
    )
    save_figure(fig, output_directory / "transverse_vortex")

    if all_lz_history:
        total_lz = np.concatenate(all_lz_history)
        t_axis = np.linspace(0, t_final_periods * laser_period, len(total_lz))
        fig = plt.figure(dpi=200)
        plot_beam_angular_momentum_evolution(
            fig,
            t_axis,
            total_lz,
            laser_period,
            title=f"Total Beam Angular Momentum (BAM) $\\mathcal{{L}}_z$ evolution ($j_z = {j_z}$)",
        )
        save_figure(fig, output_directory / "bam_evolution")
