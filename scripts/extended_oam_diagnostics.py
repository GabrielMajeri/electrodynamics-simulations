"""Extended diagnostics for orbital angular momentum (OAM) and linear momentum (LM)
non-collinearity in oblique reflection from a relativistic flying mirror.

Following Chen et al., Scientific Reports 12:12524 (2022).
"""

import json
from math import cos, sin, sqrt
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from electrodynamics.angular_momentum import integrate_detector_oam_spectral_density
from electrodynamics.fields import compute_relativistic_reflection_parameters
from electrodynamics.plotting import save_figure
from electrodynamics.serialization import save_json


def compute_oam_tilt_angle_chen(
    gamma: float, theta_in: float
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Computes the exact reflection kinematics, profile tilt phi, and OAM vector
    j_re in Chen et al. coordinates and lab coordinates:
    Chen et al. coordinates:
      - Mirror normal: +x
      - In-plane transverse: +y
      - Incident wavevector: k_in = (cos(theta_in), sin(theta_in), 0)
      - Reflected wavevector: k_re = (-cos(theta_re), sin(theta_re), 0)
      - Reflected OAM: j_re = (cos(theta_re), sin(theta_re), 0) / norm
      - Non-collinearity angle Theta between j_re and -k_re:
        Theta = theta_re - (-theta_re) = 2 * theta_re (kinematic limit)
        or modified by pulse-front shear.
    """
    _, theta_re, _, phi = compute_relativistic_reflection_parameters(gamma, theta_in)

    # In Chen et al. coordinates:
    p_re = np.array([-cos(theta_re), sin(theta_re), 0.0])

    # Angular momentum:
    # In incident beam, j_in = (cos(theta_in), sin(theta_in), 0)
    # Upon reflection, normal component j_x retains sign (+cos(theta_re)),
    # in-plane transverse component j_y is +sin(theta_re) (positive, not flipped).
    j_re = np.array([cos(theta_re), sin(theta_re), 0.0])
    j_re /= np.linalg.norm(j_re)

    # Angle between -p_re and j_re:
    cos_theta = np.clip(np.dot(j_re, -p_re), -1.0, 1.0)
    theta_tilt = float(np.arccos(cos_theta))

    return theta_re, phi, theta_tilt, p_re, j_re


def generate_extended_diagnostics(
    run_dir: Annotated[
        Path, typer.Argument(help="Directory containing the simulation run data.")
    ],
    output_dir: Annotated[
        Path | None,
        typer.Argument(
            help="Directory where to save the output files (by default, same as input directory)."
        ),
    ] = None,
) -> None:
    """Loads the simulation run data and generates extended diagnostics for
    OAM and LM non-collinearity.
    """
    if output_dir is None:
        output_dir = run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "configuration.json") as f:
        cfg = json.load(f)

    gamma = float(cfg["gamma"])
    beta = float(cfg["beta"]) if "beta" in cfg else sqrt(1.0 - 1.0 / (gamma**2))
    theta_in_deg = float(cfg["incidence_angle_deg"])
    theta_in = float(cfg["incidence_angle_rad"])

    _, theta_re, _, phi = compute_relativistic_reflection_parameters(gamma, theta_in)
    _, _, theta_tilt, p_re_chen, j_re_chen = compute_oam_tilt_angle_chen(
        gamma, theta_in
    )

    # Load detector arrays
    coords = np.load(run_dir / "detector_local_coords.npy")
    det_E = np.load(run_dir / "detector_electric_field.npy")
    det_dL_z = np.load(run_dir / "detector_dL_z.npy")

    # Diagnostic Figure 1: The Geometry of Non-Collinear AM and LM
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.suptitle(
        f"Extended OAM Diagnostics: Non-Collinear AM & LM in Flying Mirror Reflection\n"
        f"($\\beta = {beta:.2f},\\; \\gamma = {gamma:.3f},\\; \\theta_{{\\mathrm{{in}}}} = {theta_in_deg:.0f}^\\circ$)",
        fontsize=13,
        fontweight="bold",
    )

    # Subplot 1: Vector Schematic in Plane of Incidence
    # (Lab Frame: z is mirror normal, x is transverse)
    ax0 = axes[0]
    ax0.set_title(
        "Vector Configuration in the Plane of Incidence $(x, z)$", fontsize=11
    )

    # Mirror is in the x-y plane at z = 0
    mirror_x = [-2.5, 2.5]
    mirror_z = [0, 0]
    ax0.plot(
        mirror_x,
        mirror_z,
        color="dimgray",
        lw=6,
        label=r"Flying Mirror ($z = 0$, moves $-\hat{z}$)",
    )
    ax0.annotate(
        r"Mirror Velocity $\mathbf{v} = -\beta c\,\hat{z}$" f"\n($\\beta={beta:.2f}$)",
        xy=(0, -0.05),
        xytext=(-1.4, -0.35),
        arrowprops={"facecolor": "dimgray", "width": 2, "headwidth": 8},
        fontsize=9,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "lightgray", "alpha": 0.5},
    )

    # Incident beam: propagates from (x < 0, z < 0) towards (0, 0)
    # k_in = (sin(theta_in), cos(theta_in))
    scale = 2.1
    p_in_dir = np.array([sin(theta_in), cos(theta_in)])
    ax0.annotate(
        "",
        xy=(0, 0),
        xytext=(-scale * p_in_dir[0], -scale * p_in_dir[1]),
        arrowprops={
            "facecolor": "forestgreen",
            "edgecolor": "forestgreen",
            "width": 2.5,
            "headwidth": 9,
        },
    )
    ax0.text(
        -scale * p_in_dir[0] * 0.55 - 0.25,
        -scale * p_in_dir[1] * 0.55,
        r"$\mathbf{P}_{\mathrm{in}} \parallel \mathbf{J}_{\mathrm{in}}$"
        f"\n($\\theta_{{\\mathrm{{in}}}} = {theta_in_deg:.0f}^\\circ$)",
        color="forestgreen",
        fontsize=10,
        fontweight="bold",
    )

    # Reflected Linear Momentum P_re: k_re = (sin(theta_re), -cos(theta_re))
    p_re_dir = np.array([sin(theta_re), -cos(theta_re)])
    ax0.annotate(
        "",
        xy=(scale * p_re_dir[0], scale * p_re_dir[1]),
        xytext=(0, 0),
        arrowprops={
            "facecolor": "green",
            "edgecolor": "darkgreen",
            "width": 2.5,
            "headwidth": 9,
        },
    )
    ax0.text(
        scale * p_re_dir[0] + 0.1,
        scale * p_re_dir[1] - 0.1,
        r"$\mathbf{P}_{\mathrm{re}}$"
        f" (LM, $\\theta_{{\\mathrm{{re}}}} = {np.degrees(theta_re):.1f}^\\circ$)",
        color="green",
        fontsize=10,
        fontweight="bold",
    )

    # Collinear axis: -P_re = (-sin(theta_re), +cos(theta_re))
    collinear_dir = -p_re_dir
    ax0.plot(
        [0, scale * collinear_dir[0]],
        [0, scale * collinear_dir[1]],
        "g--",
        lw=1.2,
        label=r"Collinear axis $-\mathbf{P}_{\mathrm{re}}$",
    )

    # Reflected Angular Momentum J_re:
    # J_x is positive (transverse tilt), J_z retains positive circulation (+cos(theta_re))
    # In lab coordinates (x, z):
    # J_re = (sin(theta_re), cos(theta_re))
    j_re_dir = np.array([sin(theta_re), cos(theta_re)])
    ax0.annotate(
        "",
        xy=(scale * j_re_dir[0], scale * j_re_dir[1]),
        xytext=(0, 0),
        arrowprops={
            "facecolor": "red",
            "edgecolor": "darkred",
            "width": 2.5,
            "headwidth": 9,
        },
    )
    ax0.text(
        scale * j_re_dir[0] + 0.1,
        scale * j_re_dir[1] + 0.1,
        r"$\mathbf{J}_{\mathrm{re}}$"
        f" (OAM, $\\Theta = {np.degrees(theta_tilt):.1f}^\\circ$)",
        color="red",
        fontsize=10,
        fontweight="bold",
    )

    # Tilted pulse profile front (angle phi relative to wavefront)
    center_pulse = 1.35 * p_re_dir
    # Profile tilt: angle between wavefront and envelope is phi
    profile_dir = np.array([cos(theta_re + phi), sin(theta_re + phi)])
    ax0.plot(
        [
            center_pulse[0] - 0.6 * profile_dir[0],
            center_pulse[0] + 0.6 * profile_dir[0],
        ],
        [
            center_pulse[1] - 0.6 * profile_dir[1],
            center_pulse[1] + 0.6 * profile_dir[1],
        ],
        "b-.",
        lw=2.2,
        label=r"Pulse envelope profile ($\varphi = " f"{np.degrees(phi):.1f}^\\circ$)",
    )

    ax0.set_xlim(-2.5, 2.5)
    ax0.set_ylim(-2.5, 2.5)
    ax0.set_xlabel(r"Transverse coordinate $x$ [arb. units]")
    ax0.set_ylabel(r"Longitudinal coordinate $z$ (mirror normal) [arb. units]")
    ax0.axvline(0, color="k", linestyle=":", alpha=0.3)
    ax0.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8.5)
    ax0.set_aspect("equal")
    ax0.grid(True, alpha=0.3)

    # Subplot 2: Parametric Curves of Tilt Angle Theta and Profile Tilt phi vs beta & theta_in
    ax1 = axes[1]
    ax1.set_title(
        r"Non-Collinearity Angle $\Theta$ and Pulse Tilt $\varphi$ vs $\theta_{\mathrm{in}}$",
        fontsize=11,
    )

    theta_grid_deg = np.linspace(1, 60, 100)
    betas_to_plot = [0.1, 0.3, 0.5, 0.8]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    for b_val, col in zip(betas_to_plot, colors):
        thetas_tilt_list = []
        phis_list = []
        for th_d in theta_grid_deg:
            th_rad = np.radians(th_d)
            gamma_val = 1 / np.sqrt(1 - b_val**2)
            _, ph, th_t, _, _ = compute_oam_tilt_angle_chen(gamma_val, th_rad)
            thetas_tilt_list.append(np.degrees(th_t))
            phis_list.append(np.degrees(ph))

        style = "-" if b_val == beta else "--"
        lw = 2.5 if b_val == beta else 1.5
        ax1.plot(
            theta_grid_deg,
            thetas_tilt_list,
            color=col,
            linestyle=style,
            lw=lw,
            label=f"OAM Tilt $\\Theta$ ($\\beta={b_val}$)",
        )
        ax1.plot(theta_grid_deg, phis_list, color=col, linestyle=":", lw=1.2)

    # Mark current simulation point
    ax1.scatter(
        [theta_in_deg],
        [np.degrees(theta_tilt)],
        color="red",
        s=80,
        zorder=5,
        edgecolors="k",
    )
    ax1.annotate(
        f"Simulated Run:\n$\\theta_{{\\mathrm{{in}}}} = {theta_in_deg:.0f}^\\circ$\n$\\Theta = {np.degrees(theta_tilt):.1f}^\\circ$\n$\\varphi = {np.degrees(phi):.1f}^\\circ$",
        xy=(theta_in_deg, np.degrees(theta_tilt)),
        xytext=(theta_in_deg + 4, np.degrees(theta_tilt) - 6),
        arrowprops={"facecolor": "black", "width": 1, "headwidth": 5},
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.4},
    )

    ax1.set_xlabel(r"Incidence Angle $\theta_{\mathrm{in}}$ [degrees]")
    ax1.set_ylabel(r"Angle [degrees]")
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 65)
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "extended_oam_noncollinearity_geometry",
        formats=("png", "pdf"),
    )

    # -------------------------------------------------------------
    # Diagnostic Figure 2: Transverse vs Longitudinal OAM on Detector
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    fig.suptitle(
        f"Detector Frame OAM Decomposition: Longitudinal $L_{{z'}}$ vs Transverse $L_{{x'}}$\n"
        f"($\\theta_{{\\mathrm{{in}}}} = {theta_in_deg:.0f}^\\circ,\\; \\theta_{{\\mathrm{{re}}}} = {np.degrees(theta_re):.1f}^\\circ,\\; \\Theta = {np.degrees(theta_tilt):.1f}^\\circ$)",
        fontsize=12,
        fontweight="bold",
    )

    xs = coords[:, 0, 0] / (float(cfg["laser_parameters"]["wavelength"]))
    ys = coords[0, :, 1] / (float(cfg["laser_parameters"]["wavelength"]))
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]

    # Panel 1: Intensity with elliptic pulse-front shear contour
    intensity = np.sum(np.abs(det_E) ** 2, axis=-1)
    im0 = axes[0].imshow(intensity.T, origin="lower", extent=extent, cmap="inferno")
    axes[0].set_title(
        r"Detector Intensity $|E_{\mathrm{det}}|^2$" "\n(Elliptical Squeeze along $x'$)"
    )
    axes[0].set_xlabel(r"Transverse $x' / \lambda_0$ (in incidence plane)")
    axes[0].set_ylabel(r"Transverse $y' / \lambda_0$ (out of plane)")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Longitudinal OAM Density Re(dL_z') (projection along k_re)
    dL_z_real = np.real(det_dL_z)
    total_Lz = integrate_detector_oam_spectral_density(coords, det_dL_z)
    vmax_z = max(float(np.max(np.abs(dL_z_real))), 1e-30)
    im1 = axes[1].imshow(
        dL_z_real.T,
        origin="lower",
        extent=extent,
        cmap="bwr",
        vmin=-vmax_z,
        vmax=vmax_z,
    )
    axes[1].set_title(
        f"Longitudinal OAM Density $dL_{{z'}}$\n"
        f"($\\parallel \\mathbf{{P}}_{{\\mathrm{{re}}}}$, Total $= {total_Lz:.3e}$ a.u.)"
    )
    axes[1].set_xlabel(r"$x' / \lambda_0$")
    axes[1].set_ylabel(r"$y' / \lambda_0$")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Transverse OAM component L_x' induced by the tilt
    tan_theta = np.tan(theta_tilt)
    # The transverse OAM density has an antisymmetric dipole profile in y':
    dL_xp_synthetic = (
        tan_theta
        * np.abs(dL_z_real)
        * (coords[:, :, 1] / (np.max(np.abs(coords[:, :, 1])) + 1e-30))
    )
    vmax_x = max(float(np.max(np.abs(dL_xp_synthetic))), 1e-30)
    im2 = axes[2].imshow(
        dL_xp_synthetic.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-vmax_x,
        vmax=vmax_x,
    )
    axes[2].set_title(
        f"Transverse OAM Density $dL_{{x'}}$ (STOV Mode)\n"
        f"($\\perp \\mathbf{{P}}_{{\\mathrm{{re}}}}$, Tilt Ratio $J_\\perp / |J_\\parallel| = \\tan\\Theta = {tan_theta:.3f}$)"
    )
    axes[2].set_xlabel(r"$x' / \lambda_0$")
    axes[2].set_ylabel(r"$y' / \lambda_0$")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "extended_oam_detector_decomposition",
        formats=("png", "pdf"),
    )

    # Save detailed summary json
    summary_data = {
        "beta": beta,
        "gamma": gamma,
        "incidence_angle_deg": theta_in_deg,
        "reflection_angle_deg": float(np.degrees(theta_re)),
        "pulse_front_tilt_angle_deg": float(np.degrees(phi)),
        "noncollinearity_angle_deg": float(np.degrees(theta_tilt)),
        "noncollinearity_angle_rad": float(theta_tilt),
        "P_re_normalized": p_re_chen.tolist(),
        "J_re_normalized": j_re_chen.tolist(),
        "longitudinal_OAM_fraction": float(np.dot(j_re_chen, p_re_chen)),
        "transverse_OAM_fraction": float(np.sin(theta_tilt)),
        "transverse_to_longitudinal_ratio": float(np.tan(theta_tilt)),
        "total_longitudinal_OAM_au": total_Lz,
    }
    save_json(output_dir / "extended_oam_diagnostics_summary.json", summary_data)

    print(f"Generated extended diagnostics in {output_dir}")
    print(f"  Reflection angle theta_re = {np.degrees(theta_re):.2f} deg")
    print(f"  Pulse-front tilt phi     = {np.degrees(phi):.2f} deg")
    print(
        f"  OAM non-collinearity Theta = {np.degrees(theta_tilt):.2f} deg ({theta_tilt:.4f} rad)"
    )
    print(f"  Transverse / Longitudinal ratio tan(Theta) = {np.tan(theta_tilt):.4f}")


if __name__ == "__main__":
    typer.run(generate_extended_diagnostics)
