import numpy as np

from electrodynamics.beams import (
    PolarizationVector,
    compute_gaussian_beam_electric_field,
)
from electrodynamics.plotting import (
    plot_electric_field_cross_section,
    plot_electric_field_distribution,
    plot_electric_field_intensity,
)


def main() -> int:
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
    plot_electric_field_intensity(R, Z, electric_field)

    ### Gaussian beam cross-sectional profile
    plot_electric_field_cross_section(rs, zs, electric_field)

    ### Electric field distribution in the r-z plane
    plot_electric_field_distribution(R, Z, electric_field)

    return 0


if __name__ == "__main__":
    exit(main())
