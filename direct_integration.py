from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.constants import num_particles, c, omega_laser, lmbd
from electrodynamics.trajectories import simulate_circular_trajectories
from electrodynamics.typing import RealArray


def plot_configuration(trajectories: RealArray, detector_positions: RealArray) -> None:
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

    plt.tight_layout()
    plt.savefig("plots/experimental_setup.pdf")

    # plt.show()
    plt.close()


def plot_particle_trajectory(
    particle_displacements: RealArray, particle_index: int
) -> None:
    plt.title(f"$r_0(t)$ for particle #{particle_index + 1}")

    plt.plot(timestamps, particle_displacements[particle_index, :, 0], label="x")
    plt.plot(timestamps, particle_displacements[particle_index, :, 1], label="y")

    plt.xlabel("Time $t$")
    plt.ylabel("Displacement")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/particle_trajectory.pdf")
    plt.close()


if __name__ == "__main__":
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    initial_time = perf_counter()

    print("Computing electron trajectories")

    start_time = perf_counter()
    large_circle_radius = 50 * lmbd
    integration_duration, timestamps, centers, trajectories = (
        simulate_circular_trajectories(large_circle_radius)
    )
    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Simulating circular trajectories took {duration:.4g} seconds")

    # Allocate a buffer for the detectors' positions
    detector_positions = np.zeros(shape=(256, 3), dtype=np.float64)

    # Linearly displaced detector points
    detector_positions[:, 0] = np.linspace(
        -50 * large_circle_radius,
        50 * large_circle_radius,
        detector_positions.shape[0],
    )

    # Radially displaced detector points
    # phi = np.linspace(0, 2 * np.pi, 256)
    # detector_positions[:, 0] = 3.8 * large_circle_radius * np.cos(phi)
    # detector_positions[:, 1] = 3.8 * large_circle_radius * np.sin(phi)

    # Offset it to be very high in the Z direction
    detector_positions[:, 2] = 1000 * lmbd

    if True:
        plot_configuration(trajectories, detector_positions)

    # TODO: timings

    # Current offset of particle from "center" of its motion, aka r_0(t)
    particle_displacements = trajectories - centers[:, np.newaxis, :]

    if True:
        plot_particle_trajectory(particle_displacements, 0)

    # print("x shape:", detector_positions.shape)
    # print("r(t) shape:", particle_displacements.shape)

    # Compute relative trajectories (with respect to each detector positions)
    # R(x_0, t) = x_0 - r_0(t) = (x - R_0) - (r(t) - R_0) = x - r(t)
    Rs = (
        detector_positions[:, np.newaxis, np.newaxis, :]
        - particle_displacements[np.newaxis, :, :, :]
    )

    # print("R(x_0, t) shape:", Rs.shape)

    R_norms = np.linalg.vector_norm(Rs, axis=-1)

    # Compute relative directions
    # n(x_0, t) = R(x_0, t)/|R(x_0, t)|
    ns = Rs / np.expand_dims(R_norms, -1)

    # print("n(x_0, t) shape:", ns.shape)

    print()
    print("Checking for faster-than-light particles...")

    particle_velocities = np.gradient(particle_displacements, timestamps, axis=1)
    assert np.all(particle_velocities <= c), "Particles are moving faster than light!"

    relativistic_factor = particle_velocities / c

    # Construct a cutoff
    slope = 20
    cutoff = np.exp(-1 / (timestamps / slope + 1e-10) ** 2) * np.exp(
        -1 / (((integration_duration - timestamps) / slope + 1e-10) ** 2)
    )

    # The frequency we are interested in (for which we are computing the Fourier transform)
    frequency = omega_laser

    print("Computing oscillatory kernel (complex exponential)")

    # exp(i * omega * (t + R(x_0, t)/c))
    oscillatory_kernel = np.exp(1j * frequency * (timestamps + R_norms / c))

    print("Computing first term of the integrand (1/|R|)")

    # ((i * omega)/c) * (beta(t) - n(x_0, t) * (dot(n, beta)))/|R(x_0, t)|
    integrand_first_term = (
        ((1j * frequency) / c)
        * (
            relativistic_factor
            - ns * np.expand_dims(np.linalg.vecdot(ns, relativistic_factor), axis=-1)
        )
        / np.expand_dims(R_norms, axis=-1)
    )

    # print("Computing second term of the integrand (1/|R|^2)")

    # # n(x_0, t)/|R(x_0, t)|^2
    # integrand_second_term = ns / np.expand_dims(np.square(R_norms), axis=-1)

    print("Summing up results (computing Riemann integral)")

    dt = timestamps[1] - timestamps[0]

    # First, sum up along time dimension, to compute integral
    oscillatory_kernel = np.expand_dims(oscillatory_kernel, axis=-1)
    first_term = dt * np.sum(
        oscillatory_kernel * integrand_first_term,
        # * cutoff[np.newaxis, np.newaxis, :, np.newaxis],
        axis=-2,
    )
    # TODO: compute second term using alternative formula
    # second_term = dt * np.sum(
    #     oscillatory_kernel * integrand_second_term,
    #     # * cutoff[np.newaxis, np.newaxis, :, np.newaxis],
    #     axis=-2,
    # )

    # Now sum up each particle's contribution
    first_term = np.sum(first_term, axis=-2)
    # second_term = np.sum(second_term, axis=-2)

    final_time = perf_counter()
    total_duration = final_time - initial_time
    print(f"Execution took a total of {total_duration:.3g} seconds")

    print("Plotting results")

    # Plot terms' magnitude

    first_term_norms = np.linalg.vector_norm(first_term, axis=-1)
    # second_term_norms = np.linalg.vector_norm(second_term, axis=-1)

    fig, ax = plt.subplots(1, 1)  # , figsize=(8, 5))

    plt.suptitle("Electric field spectral intensity $|E(x_0, \\omega)|$")

    ax.set_title("First-order term ($1/|R|$)")
    ax.plot(detector_positions[:, 0], first_term_norms)
    ax.grid()
    ax.set_xlabel("Detector position")
    ax.set_ylabel("Electric vector field norm")

    # axes[1].set_title("Second term ($1/|R|^2$)")
    # axes[1].plot(second_term_norms, color="orange")
    # axes[1].grid()
    # axes[1].set_xlabel("Detector position")
    # axes[1].set_ylabel("Electric vector field norm")

    plt.tight_layout()
    fig.savefig("plots/electric_field_magnitudes.pdf")
    plt.close()

    exit(0)

    r_0s = particle_displacements

    # Compute detector displacement (in each particle frame of reference)
    x_0s = detector_positions[:, np.newaxis, :] - centers[np.newaxis, :, :]
    x_0s_norms = np.linalg.vector_norm(x_0s, axis=-1)

    n_0s = x_0s / x_0s_norms[:, :, np.newaxis]

    if False:
        frequency = omega_laser * 1.00
        n_0_dot_r_0 = np.vecdot(n_0s[0], r_0s[0])
        exponent = frequency * timestamps - frequency / c * n_0_dot_r_0

        if False:
            plt.title("$\\frac{d}{dt} \\left(n_0 \\cdot r_0\\right) (t)$")

            # TODO: Need to include timestamps for dt!!
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
        # TODO: Need to include timestamps for dt!!
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
        # TODO: Need to include timestamps for dt!!
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

    print("Computing (n_0, r_0(t)) dot products")

    start_time = perf_counter()

    n_0s_dot_r_0s = np.vecdot(n_0s[:, :, np.newaxis, :], r_0s)

    g = frequency * timestamps - frequency / c * n_0s_dot_r_0s

    exponent = 1j * g

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing exponents took {duration:.4g} seconds")

    start_time = perf_counter()

    n_0s_dot_v_0s = np.gradient(n_0s_dot_r_0s, axis=-1)

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing velocities took {duration:.4g} seconds")

    start_time = perf_counter()

    oscillatory_kernel = np.exp(exponent)
    integrand = oscillatory_kernel * n_0s_dot_v_0s

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing integrands took {duration:.4g} seconds")

    start_time = perf_counter()

    # Approximate integral using Riemann sum
    result = (1 / x_0s_norms) * dt * np.sum(integrand, axis=-1)

    # Sum across particles
    result = np.sum(result, axis=-1)

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing integrals using Riemann sums took {duration:.4g} seconds")

    final_time = perf_counter()
    total_duration = final_time - initial_time
    print(f"Execution took a total of {total_duration:.3g} seconds")

    print("Plotting results")

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
