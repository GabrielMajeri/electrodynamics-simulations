from argparse import ArgumentParser
from sys import exit

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.beams import (
    PolarizationVector,
    compute_electric_and_magnetic_field_for_laguerre_gauss_beam,
    compute_gaussian_beam_electric_field,
)
from electrodynamics.constants import lmbd
from electrodynamics.typing import RealArray


def plot_gaussian_beam_electric_field_intensity(
    R: RealArray, Z: RealArray, intensity: RealArray
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_title("Gaussian beam profile")

    ax.plot_surface(
        R,
        Z,
        intensity,
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


def plot_gaussian_beam_electric_field_distribution(
    R: RealArray, Z: RealArray, electric_field: RealArray
) -> None:
    fig = plt.figure(dpi=200)

    ax = fig.add_subplot()

    ax.set_title("Gaussian beam electric field distribution")

    ax.pcolormesh(Z, R, np.real(electric_field), vmin=-1.0, vmax=1.0, cmap="coolwarm")

    ax.set_xlabel("$z$")
    ax.set_ylabel("$r$")

    ax.grid()

    fig.tight_layout()
    fig.savefig("plots/gaussian_beam_electric_field_distribution.png")


def plot_gaussian_beam_cross_section_intensity(
    rs: RealArray, zs: RealArray, intensity: RealArray
) -> None:
    fig = plt.figure()

    ax = fig.add_subplot()

    ax.set_title("Gaussian beam cross-section at $z = 0$")

    z_0_index = len(zs) // 2

    ax.plot(rs, intensity[:, z_0_index])

    ax.set_xlabel("$R$")
    ax.set_ylabel("$|I|$")

    ax.grid()

    fig.tight_layout()
    fig.savefig("plots/gaussian_beam_cross_section.pdf")


def plot_laguerre_gauss_beam_electric_field_cross_section(
    xs: RealArray,
    ys: RealArray,
    electric_field: RealArray,
    azimuthal_index: int,
    radial_index: int,
) -> None:
    fig = plt.figure(dpi=200)

    ax = fig.add_subplot()

    ax.set_title(
        f"Laguerre-Gauss beam cross-section at $z = 0$\n$l = {azimuthal_index}$, $p = {radial_index}$"
    )

    mesh = ax.pcolormesh(xs, ys, electric_field, antialiased=True, cmap="Blues")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.grid()
    ax.set_aspect(1)

    fig.colorbar(mesh)

    fig.tight_layout()
    fig.savefig("plots/laguerre_gauss_beam_cross_section.png")


def main() -> int:
    parser = ArgumentParser(
        "plot_beams.py",
        description="Makes various plots for a given type of laser beam.",
    )

    parser.add_argument(
        "--beam-type", default="laguerre-gauss", choices=["gaussian", "laguerre-gauss"]
    )

    args = parser.parse_args()

    beam_type = args.beam_type
    if beam_type == "gaussian":
        print("Making plots for Gaussian beam")

        rs = np.linspace(-25, 25, 512)
        zs = np.linspace(-100, 100, 512)

        R, Z = np.meshgrid(rs, zs, sparse=False)

        # E_0
        amplitude = 1.0

        # Lambda
        wavelength = 3

        electric_field = compute_gaussian_beam_electric_field(
            R, Z, amplitude, wavelength
        )

        electric_field_phase = np.real(electric_field)
        intensity = np.abs(electric_field)

        ### Electric field surface plot
        plot_gaussian_beam_electric_field_intensity(R, Z, intensity)

        ### Gaussian beam cross-sectional intensity profile
        plot_gaussian_beam_cross_section_intensity(rs, zs, intensity)

        ### Electric field distribution in the r-z plane
        plot_gaussian_beam_electric_field_distribution(R, Z, electric_field_phase)

    elif beam_type == "laguerre-gauss":
        print("Making plots for Laguerre-Gauss beam")

        nx = 256
        ny = 256

        xs = np.linspace(-4e6, 4e6, nx)
        ys = np.linspace(-4e6, 4e6, ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij", sparse=False)

        positions = np.vstack((X.ravel(), Y.ravel())).T
        positions = np.concatenate(
            (positions, np.zeros((positions.shape[0], 1), dtype=np.float64)), axis=1
        )

        wavelength = lmbd
        waist_radius = 75 * wavelength
        radial_index = 2
        azimuthal_index = -2
        polarization = PolarizationVector(1, 0)
        time = 0

        electric_field, _magnetic_field = (
            compute_electric_and_magnetic_field_for_laguerre_gauss_beam(
                amplitude=1,
                waist_radius=waist_radius,
                wavelength=wavelength,
                radial_index=radial_index,
                azimuthal_index=azimuthal_index,
                polarization=polarization,
                positions=positions,
                time=time,
            )
        )

        plot_laguerre_gauss_beam_electric_field_cross_section(
            positions[:, 0].reshape(nx, ny),
            positions[:, 1].reshape(nx, ny),
            # np.real(electric_field[:, 1]).reshape(nx, ny),
            np.linalg.vector_norm(electric_field, axis=-1).reshape(nx, ny),
            azimuthal_index,
            radial_index,
        )

    else:
        print(f"Unknown beam type: {beam_type}")

        return 1

    return 0


if __name__ == "__main__":
    exit(main())
