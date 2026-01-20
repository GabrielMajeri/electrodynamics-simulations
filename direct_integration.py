from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


type Array = npt.NDArray[np.float64]


### Physical constants (in atomic units)
# Speed of light
c = 137.036

# Angular frequency of laser radiation (we assume it's monochromatic)
omega_laser = 0.057

# Wavelength of laser radiation
lmbd = c * (2 * np.pi / omega_laser)
# print("Lambda:", lmbd)

# Number of particles (electrons) to simulate
num_particles = 30

particle_indices = np.arange(num_particles, dtype=np.int32)


class CircularTrajectories(NamedTuple):
    integration_duration: float
    timestamps: Array
    centers: Array
    trajectories: Array


def simulate_circular_trajectories(
    # Radius of the large circle around which the particle's center are positioned.
    large_circle_radius,
    # How many timestamps to use (will affect trajectory sampling frequency)
    num_timestamps=4096,
) -> CircularTrajectories:
    """Code to numerically compute the particle's trajectories.

    They move around in a small circular trajectory around a center of motion,
    and these centers are radially distributed around the origin.
    """

    # Determine the particles' center-of-motion
    centers = np.array(
        (
            large_circle_radius * np.cos(2 * np.pi * particle_indices / num_particles),
            large_circle_radius * np.sin(2 * np.pi * particle_indices / num_particles),
            np.zeros(num_particles),
        )
    ).T

    trajectories = np.empty((num_particles, num_timestamps, 3))

    # Determine for how long the simulation/numerical integration will run
    # We'll use an integer multiple of the laser pulse's period
    num_periods = 40
    integration_duration = num_periods * (2 * np.pi) / omega_laser
    timestamps = np.linspace(0, integration_duration, num_timestamps)

    # We apply a decay to the trajectories to ensure the integral decays near the boundaries
    slope = 20
    cutoff = np.exp(-1 / (timestamps / slope + 1e-10) ** 2) * np.exp(
        -1 / (((integration_duration - timestamps) / slope + 1e-10) ** 2)
    )

    trajectory_radius = 0.75 * lmbd * cutoff

    # Compute instantaneous velocities and make sure we don't go faster than the speed of light
    # (this might happen if we set a trajectory with too large a circumference)
    assert np.all(trajectory_radius / (2 * np.pi / omega_laser) < c), (
        "Particles move faster than speed of light"
    )

    num_wraps = 1
    phi_0 = particle_indices * num_wraps * (2 * np.pi) / num_particles

    trajectories[:, :, 0] = centers[:, 0, np.newaxis] + (
        trajectory_radius
        * np.cos(omega_laser * timestamps[np.newaxis, :] - phi_0[:, np.newaxis])
    )

    trajectories[:, :, 1] = centers[:, 1, np.newaxis] + (
        trajectory_radius
        * np.sin(omega_laser * timestamps[np.newaxis, :] - phi_0[:, np.newaxis])
    )

    trajectories[:, :, 2] = 0

    return CircularTrajectories(integration_duration, timestamps, centers, trajectories)


def plot_configuration(trajectories: Array, detector_positions: Array) -> None:
    "Plot the current simulated setup (particles and detector)."

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for particle in range(num_particles):
        x, y, z = (
            trajectories[particle, :, 0],
            trajectories[particle, :, 1],
            trajectories[particle, :, 2],
        )
        ax.plot(
            x,
            y,
            z,
        )

    ax.plot(*detector_positions.T)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.savefig("plots/experimental_setup.pdf")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    large_circle_radius = 50 * lmbd
    integration_duration, timestamps, centers, trajectories = (
        simulate_circular_trajectories(large_circle_radius)
    )

    # Allocate a buffer for the detectors' positions
    detector_positions = np.zeros(shape=(1024, 3), dtype=np.float64)

    # Linearly displaced detector points
    detector_positions[:, 0] = np.linspace(
        -30 * large_circle_radius,
        30 * large_circle_radius,
        detector_positions.shape[0],
    )

    # Radially displaced detector points
    # phi = np.linspace(0, 2 * np.pi, 1024)
    # detector_positions[:, 0] = 3.8 * large_circle_radius * np.cos(phi)
    # detector_positions[:, 1] = 3.8 * large_circle_radius * np.sin(phi)

    detector_positions[:, 2] = 1000 * lmbd

    if True:
        plot_configuration(trajectories, detector_positions)

    # Current offset of particle from "center" of its motion
    particle_displacements = trajectories - centers[:, np.newaxis, :]
    r_0s = particle_displacements

    if True:
        plt.title("$r_0(t)$ for particle #1")
        plt.plot(timestamps, particle_displacements[0, :, 0], label="x")
        plt.plot(timestamps, particle_displacements[0, :, 1], label="y")
        plt.legend()
        plt.grid()
        plt.savefig("plots/particle_trajectory.pdf")
        plt.close()

    # Compute detector displacement (in each particle frame of reference)
    x_0s = detector_positions[:, np.newaxis, :] - centers[np.newaxis, :, :]
    x_0s_norms = np.linalg.vector_norm(x_0s, axis=-1)

    if False:
        # Inverse of the distance between the particle's center of motion and the detector
        # We expand in a Taylor series in terms of this value (should be very small)
        print("1/|x_0|:", 1 / x_0s_norms[0])

    n_0s = x_0s / x_0s_norms[:, :, np.newaxis]

    # print(n_0s[0])

    if False:
        frequency = omega_laser * 1.00
        n_0_dot_r_0 = np.vecdot(n_0s[0], r_0s[0])
        exponent = frequency * timestamps - frequency / c * n_0_dot_r_0

        if False:
            plt.title("$\\frac{d}{dt} \\left(n_0 \\cdot r_0\\right) (t)$")

            plt.plot(timestamps, np.gradient(n_0_dot_r_0), label="Derivative")
            plt.axhline(
                c,
                xmin=0,
                xmax=integration_duration,
                color="orange",
                label="Speed of light",
            )

            plt.grid()
            plt.legend()

            plt.xlabel("$t$")
            plt.ylabel("Derivative")

            plt.savefig("plots/derivative_of_n_0_dot_r_0.pdf")
            plt.close()

        # Check again that the exponent doesn't go to 0
        assert np.all(np.abs(np.gradient(n_0_dot_r_0, axis=-1)) < c)

        if False:
            # Plot imaginary part of the exponent of the oscillatory kernel
            plt.title("Exponent ($g(t)$)")
            plt.plot(timestamps, exponent)
            plt.grid()
            plt.savefig("plots/g.pdf")
            plt.close()

        if False:
            # Plot the derivative of the exponent
            plt.title("$g'(t)$")
            plt.plot(timestamps, np.gradient(exponent))
            plt.grid()
            plt.savefig("plots/dg_dt.pdf")
            plt.close()

        oscillatory_kernel = np.exp(1j * exponent)

        if False:
            plt.title("Oscillatory kernel")
            plt.plot(timestamps, np.angle(oscillatory_kernel), marker=".", linewidth=0)
            plt.xlabel("$t$")
            plt.ylabel("")
            plt.grid()
            plt.savefig("plots/exp_i_g.pdf")
            plt.close()

        # The integrand is now the oscillatory kernel times the derivative of the position term
        integrand = oscillatory_kernel * np.gradient(n_0_dot_r_0)

        if False:
            plt.title("Integrand")
            plt.plot(timestamps, np.abs(integrand), marker=".", linewidth=0)
            plt.xlabel("$t$")
            plt.ylabel("")
            plt.grid()
            plt.savefig("plots/integrand.pdf")
            plt.close()

        # TODO: use an integration method specialized for highly-oscillatory integrals

        dt = timestamps[1] - timestamps[0]
        # integral = (1 / x_0s_norms[0]) * dt * np.sum(np.exp(1j * exponent))
        integral = dt * np.sum(integrand)
        print("Integral value:", integral)
        print("Integral absolute value:", np.abs(integral))

    dt = timestamps[1] - timestamps[0]

    frequency = omega_laser

    n_0s_dot_r_0s = np.vecdot(n_0s[:, :, np.newaxis, :], r_0s)
    g = frequency * timestamps - frequency / c * n_0s_dot_r_0s

    exponent = 1j * g
    oscillatory_kernel = np.exp(exponent)
    integrand = oscillatory_kernel * np.gradient(n_0s_dot_r_0s, axis=0)

    # Approximate integral using Riemann sum
    result = (1 / x_0s_norms) * dt * np.sum(integrand, axis=-1)

    # Sum across particles
    result = np.sum(result, axis=-1)

    plt.title("Integration results")

    plt.plot(detector_positions[:, 0], np.abs(result), label="$|\\phi(x_0)|$")
    # plt.plot(phi, np.angle(result), label="$\\arg \\phi(x_0)$")

    plt.xlabel("$x_0$")
    plt.ylabel("Scalar potential absolute value ($|\\phi|$)")

    # plt.xlabel("$\\phi$")
    # plt.ylabel("Scalar potential argument ($\\arg \\phi$)")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/scalar_potential_vs_x_0.pdf")
    plt.close()
