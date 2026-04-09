from argparse import ArgumentParser
from sys import exit

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.beams import (
    PolarizationVector,
    compute_electric_and_magnetic_field_for_laguerre_gauss_beam,
    compute_gaussian_beam_electric_field,
)
from electrodynamics.typing import RealArray


def plot_gaussian_beam_electric_field_intensity(
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


def plot_gaussian_beam_electric_field_cross_section(
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
        "--beam-type", default="gaussian", choices=["gaussian", "laguerre-gauss"]
    )

    args = parser.parse_args()

    beam_type = args.beam_type
    if beam_type == "gaussian":
        rs = np.linspace(-25, 25, 512)
        zs = np.linspace(-100, 100, 512)

        R, Z = np.meshgrid(rs, zs, sparse=False)

        # E_0
        amplitude = 1.0

        # Lambda
        wavelength = 3

        polarization = PolarizationVector(1.0, 0.0)

        electric_field = compute_gaussian_beam_electric_field(
            R, Z, amplitude, wavelength, polarization
        )

        electric_field = np.real(electric_field[0]).squeeze()

        ### Electric field surface plot
        plot_gaussian_beam_electric_field_intensity(R, Z, electric_field)

        ### Gaussian beam cross-sectional profile
        plot_gaussian_beam_electric_field_cross_section(rs, zs, electric_field)

        ### Electric field distribution in the r-z plane
        plot_gaussian_beam_electric_field_distribution(R, Z, electric_field)

    elif beam_type == "laguerre-gauss":
        nx = 128
        ny = 128

        xs = np.linspace(-5, 5, nx)
        ys = np.linspace(-5, 5, ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij", sparse=False)

        positions = np.vstack((X.ravel(), Y.ravel())).T
        positions = np.concatenate(
            (positions, np.zeros((positions.shape[0], 1), dtype=np.float64)), axis=1
        )

        time = 0
        azimuthal_index = 2
        radial_index = 3

        electric_field, _magnetic_field = (
            compute_electric_and_magnetic_field_for_laguerre_gauss_beam(
                positions, time, azimuthal_index, radial_index
            )
        )

        plot_laguerre_gauss_beam_electric_field_cross_section(
            positions[:, 0].reshape(nx, ny),
            positions[:, 1].reshape(nx, ny),
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
