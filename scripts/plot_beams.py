import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from electrodynamics.typing import ComplexArray, RealArray


class PlaneWave:
    amplitude: float
    wave_vector: RealArray

    def __init__(
        self, amplitude: float = 1.0, wave_vector: RealArray = np.array([1.0, 0.0, 0.0])
    ) -> None:
        assert amplitude > 0, "Amplitude must be a positive real number"
        self.amplitude = amplitude

        assert wave_vector.shape == (3,), "Wave vector must be a 3D vector"
        self.wave_vector = wave_vector

    def evaluate(self, points: RealArray) -> RealArray:
        # E = E_0 * cos(dot(k, x))
        return self.amplitude * np.cos(np.linalg.vecdot(self.wave_vector, points))


def plot_plane_wave() -> None:
    dx = 0.05
    x = np.arange(-10, 10, dx, dtype=np.float64)
    x = x.reshape(len(x), 1)
    points = np.hstack((x, np.zeros((len(x), 2), dtype=np.float64)))

    wave_length = 2
    wave_vector = np.array([2 * np.pi / wave_length, 0, 0], dtype=np.float64)

    plane_wave = PlaneWave(wave_vector=wave_vector)

    E_x = plane_wave.evaluate(points)

    fig, ax = plt.subplots()

    ax.plot(x, E_x)

    ax.set_title("Electric field for plane wave travelling in $x$ direction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$E_x$")

    ax.grid(True)

    fig.savefig("plots/plane_wave.pdf")


class GaussianBeam:
    amplitude: float
    waist_radius: float
    rayleigh_length: float
    wavenumber: float

    def __init__(
        self,
        # E_0
        amplitude: float = 1.0,
        # w_0
        waist_radius: float = 1.0,
        # lambda
        wavelength: float = 1.0,
        # n
        transmission_medium_refractive_index: float = 1.0,
    ) -> None:
        assert amplitude > 0, "Amplitude must be a positive real number"
        self.amplitude = amplitude

        self.waist_radius = waist_radius

        # w_R
        self.rayleigh_length = (
            np.pi * np.square(waist_radius) * transmission_medium_refractive_index
        ) / wavelength

        # k
        self.wavenumber = (
            2 * np.pi * transmission_medium_refractive_index
        ) / wavelength

    def compute_fwhm(self, z: RealArray) -> RealArray:
        # FWHM in terms of z
        return self.waist_radius * np.sqrt(1 + np.square(z / self.rayleigh_length))

    def compute_radius_of_curvature(self, z: RealArray) -> RealArray:
        # Radius of curvature (in terms of z)
        zero_z = np.isclose(0, z)
        masked_z = np.where(zero_z, 1, z)
        return np.where(
            zero_z, +np.inf, z * (1 + np.square(self.rayleigh_length / masked_z))
        )

    def evaluate(self, z: RealArray, r: RealArray) -> ComplexArray:
        w_z = self.compute_fwhm(z)
        R_z = self.compute_radius_of_curvature(z)

        # Gouy phase
        psi_z = np.arctan(z / self.rayleigh_length)

        amplitude_factor = (
            self.amplitude * (self.waist_radius / w_z) * np.exp(-np.square(r / w_z))
        ).astype(np.complex128)
        phase_factor = np.exp(
            -1j
            * (
                self.wavenumber * z
                + self.wavenumber * (np.square(r) / (2 * R_z))
                - psi_z
            )
        ).astype(np.complex128)

        return amplitude_factor * phase_factor


def plot_gaussian_beam_cross_section(gaussian_beam: GaussianBeam) -> None:
    r = np.arange(-4, 4, 0.01)

    E = gaussian_beam.evaluate(np.asarray(0.0), r)
    E = np.abs(np.real(E))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(r, E)
    ax.fill_between(r, np.zeros_like(E), E)

    ax.set_title("Gaussian beam cross-sectional intensity")

    ax.set_xlabel("$r$")
    ax.set_ylabel("$\\vert E \\vert$")

    ax.grid(True)

    fig.savefig("plots/gaussian_beam_cross_section.pdf")


def plot_gaussian_beam_width(gaussian_beam: GaussianBeam) -> None:
    z = np.arange(-10, 10, 0.1)
    widths = gaussian_beam.compute_fwhm(z)

    fig, ax = plt.subplots(figsize=(9, 5))

    color = "tab:blue"
    ax.plot(z, widths, color=color)
    ax.plot(z, -widths, color=color)

    ax.set_title("Gaussian beam width $w(z)$")

    ax.set_xlabel("$z$")
    ax.set_ylabel("$w(z)$")

    ax.grid(True)

    fig.savefig("plots/gaussian_beam_width.pdf")


def plot_gaussian_beam_intensity_profile(gaussian_beam: GaussianBeam) -> None:
    left, right = -10, 10
    bottom, top = -5, 5
    z = np.arange(left, right, 0.01)
    r = np.arange(bottom, top, 0.01)

    Z, R = np.meshgrid(z, r)

    E = gaussian_beam.evaluate(Z, R)
    E = np.abs(np.real(E))

    fig, ax = plt.subplots(figsize=(9, 5))

    _image = ax.imshow(E, cmap="jet", extent=(left, right, bottom, top))
    # fig.colorbar(image)

    ax.set_title("Gaussian beam intensity")

    ax.set_xlabel("z")
    ax.set_ylabel("r")

    fig.savefig("plots/gaussian_beam_intensity_profile.pdf")


def plot_gaussian_beam() -> None:
    gaussian_beam = GaussianBeam()

    plot_gaussian_beam_cross_section(gaussian_beam)
    plot_gaussian_beam_width(gaussian_beam)
    plot_gaussian_beam_intensity_profile(gaussian_beam)


class LaguerreGaussBeam:
    amplitude: float
    waist_radius: float
    rayleigh_length: float
    wavenumber: float
    radial_index: int
    azimuthal_index: int

    def __init__(
        self,
        # E_0
        amplitude: float = 1.0,
        # w_0
        waist_radius: float = 1.0,
        # lambda
        wavelength: float = 1.0,
        # n
        transmission_medium_refractive_index: float = 1.0,
        # p
        radial_index: int = 1,
        # l
        azimuthal_index: int = 0,
    ) -> None:
        assert amplitude > 0, "Amplitude must be a positive real number"
        self.amplitude = amplitude

        self.waist_radius = waist_radius

        # w_R
        self.rayleigh_length = (
            np.pi * np.square(waist_radius) * transmission_medium_refractive_index
        ) / wavelength

        # k
        self.wavenumber = (
            2 * np.pi * transmission_medium_refractive_index
        ) / wavelength

        assert radial_index >= 0, "Radial index must be a non-negative integer"
        self.radial_index = radial_index

        self.azimuthal_index = azimuthal_index

    def compute_fwhm(self, z: RealArray) -> RealArray:
        # FWHM in terms of z
        return self.waist_radius * np.sqrt(1 + np.square(z / self.rayleigh_length))

    def compute_radius_of_curvature(self, z: RealArray) -> RealArray:
        # Radius of curvature (in terms of z)
        zero_z = np.isclose(0, z)
        masked_z = np.where(zero_z, 1, z)
        return np.where(
            zero_z, +np.inf, z * (1 + np.square(self.rayleigh_length / masked_z))
        )

    def evaluate(self, positions: RealArray) -> ComplexArray:
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        r = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y, x)

        abs_l = abs(self.azimuthal_index)

        w_z = self.compute_fwhm(z)
        r_over_w_z_squared = np.square(r / w_z)
        R_z = self.compute_radius_of_curvature(z)

        # Gouy phase
        psi_z = np.arctan(z / self.rayleigh_length)

        amplitude_factor = (
            self.amplitude
            * sp.special.poch((self.radial_index + 1), abs_l)
            * (self.waist_radius / w_z)
            * np.pow(np.sqrt(2) * r / w_z, abs_l)
            * sp.special.hyp1f1(-self.radial_index, abs_l + 1, 2 * r_over_w_z_squared)
            * np.exp(-r_over_w_z_squared)
        ).astype(np.complex128)
        phase_factor = np.exp(
            -1j
            * (
                self.azimuthal_index * phi
                + self.wavenumber * z
                + self.wavenumber * (np.square(r) / (2 * R_z))
                - (2 * self.radial_index + self.azimuthal_index + 1) * psi_z
            )
        ).astype(np.complex128)

        return amplitude_factor * phase_factor


def plot_laguerre_gauss_beam_cross_section(beam: LaguerreGaussBeam) -> None:
    left, right = -5, 5
    bottom, top = -5, 5
    x = np.arange(left, right, 0.01)
    y = np.arange(bottom, top, 0.01)
    z = 0

    X, Y, Z = np.meshgrid(x, y, z)
    positions = np.concatenate((X, Y, Z), axis=-1)
    positions = positions.reshape(-1, 3)

    E = beam.evaluate(positions)
    E = np.abs(E)

    E = E.reshape(len(x), len(y))

    fig, ax = plt.subplots()

    ax.imshow(E, cmap="plasma", extent=(left, right, bottom, top))

    ax.set_title(
        f"Laguerre-Gauss beam cross-sectional profile\n$p = {beam.radial_index}$, $l = {beam.azimuthal_index}$"
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.savefig("plots/laguerre_gauss_beam_cross_section.pdf")


def plot_laguerre_gauss_beam() -> None:
    beam = LaguerreGaussBeam(radial_index=2, azimuthal_index=2)

    plot_laguerre_gauss_beam_cross_section(beam)


def main() -> None:
    plot_plane_wave()

    plot_gaussian_beam()

    plot_laguerre_gauss_beam()


if __name__ == "__main__":
    main()
