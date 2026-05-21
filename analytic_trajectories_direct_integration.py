from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.constants import num_particles, c, omega_laser, lmbd
from electrodynamics.trajectories import simulate_circular_trajectories
from electrodynamics.typing import RealArray


def initialize_detector_positions(large_circle_radius: float) -> RealArray:
    num_detector_points: int = 512

    # Allocate a buffer for the detectors' positions
    detector_positions = np.zeros(shape=(num_detector_points, 3), dtype=np.float64)

    # Linearly displaced detector points
    detector_width = 50 * large_circle_radius
    detector_positions[:, 0] = np.linspace(
        -detector_width,
        detector_width,
        detector_positions.shape[0],
    )

    # Radially displaced detector points
    # phi = np.linspace(0, 2 * np.pi, num_detector_points)
    # detector_positions[:, 0] = 3.8 * large_circle_radius * np.cos(phi)
    # detector_positions[:, 1] = 3.8 * large_circle_radius * np.sin(phi)

    # Offset it to be very high in the Z direction
    detector_positions[:, 2] = 1000 * lmbd

    return detector_positions


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
    timestamps: RealArray, particle_displacements: RealArray, particle_index: int
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


def main() -> None:
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    initial_time = perf_counter()

    print("Computing electron trajectories")

    large_circle_radius = 50 * lmbd
    print(f"Initial electron positions are on a circle of radius {large_circle_radius}")

    start_time = perf_counter()
    integration_duration, timestamps, centers, trajectories, velocities = (
        simulate_circular_trajectories(large_circle_radius)
    )
    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Simulating circular trajectories took {duration:.4g} seconds")

    detector_positions = initialize_detector_positions(large_circle_radius)

    if True:
        print("Plotting experimental setup")
        plot_configuration(trajectories, detector_positions)

    # # Offset of detector point from "center" of particle motion, aka x_0(t)
    # detector_displacements = (
    #     detector_positions[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # )

    if True:
        print("Plotting trajectory of a single particle")

        # Current offset of particle from "center" of its motion, aka r_0(t)
        particle_displacements = trajectories - centers[:, np.newaxis, :]

        plot_particle_trajectory(timestamps, particle_displacements, 0)

    print("Computing relative trajectories/displacements (R_0(x_0, t))")
    start_time = perf_counter()

    # Compute relative trajectories (with respect to each detector positions)
    # R(x_0, t) = x_0 - r_0(t) = (x - R_0) - (r(t) - R_0) = x - r(t)
    Rs = (
        detector_positions[:, np.newaxis, np.newaxis, :]
        - trajectories[np.newaxis, :, :, :]
    )

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing relative displacements took {duration:.2f} seconds")

    print("Computing displacement norms (|R_0(x_0, t)|)")
    start_time = perf_counter()

    R_norms = np.linalg.vector_norm(Rs, axis=-1)

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing displacement norms took {duration:.4f} seconds")

    print("Computing relative direction vectors (n(x_0, t))")
    start_time = perf_counter()

    # Compute relative directions
    # n(x_0, t) = R(x_0, t)/|R(x_0, t)|
    ns = Rs / np.expand_dims(R_norms, -1)

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing relative direction vectors took {duration:.4f} seconds")

    print()
    print("Checking for faster-than-light particles...")

    assert np.all(velocities <= c), "Particles are moving faster than light!"

    relativistic_factor = velocities / c

    # Construct a cutoff
    slope = 20
    cutoff = np.exp(-1 / (timestamps / slope + 1e-10) ** 2) * np.exp(
        -1 / (((integration_duration - timestamps) / slope + 1e-10) ** 2)
    )

    # The frequency we are interested in (for which we are computing the Fourier transform)
    frequency = omega_laser

    print("Computing oscillatory kernel (complex exponential)")
    start_time = perf_counter()

    # exp(i * omega * (t + R(x_0, t)/c))
    oscillatory_kernel = np.exp(1j * frequency * (timestamps + R_norms / c))

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Evaluating oscillatory kernel took {duration:.4f} seconds")

    print("Computing first-order term of the integrand (1/|R|)")
    start_time = perf_counter()

    # ((i * omega)/c) * (beta(t) - n(x_0, t) * (dot(n, beta)))/|R(x_0, t)|
    integrand_first_term = (
        ((1j * frequency) / c)
        * (
            relativistic_factor
            - ns * np.expand_dims(np.linalg.vecdot(ns, relativistic_factor), axis=-1)
        )
        / np.expand_dims(R_norms, axis=-1)
    )

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing first-order term took {duration:.4f} seconds")

    # print("Computing second term of the integrand (1/|R|^2)")

    # # n(x_0, t)/|R(x_0, t)|^2
    # integrand_second_term = ns / np.expand_dims(np.square(R_norms), axis=-1)

    print("Summing up results (computing Riemann integral)")
    start_time = perf_counter()

    dt = timestamps[1] - timestamps[0]

    # First, sum up along time dimension, to compute integral
    oscillatory_kernel = np.expand_dims(oscillatory_kernel, axis=-1)
    first_term = dt * np.sum(
        oscillatory_kernel
        * integrand_first_term
        * cutoff[np.newaxis, np.newaxis, :, np.newaxis],
        axis=-2,
    )
    # TODO: compute second term using alternative formula
    # second_term = dt * np.sum(
    #     oscillatory_kernel * integrand_second_term,
    #     # * cutoff[np.newaxis, np.newaxis, :, np.newaxis],
    #     axis=-2,
    # )

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing Riemann sums took {duration:.4f} seconds")

    print("Summing up particles' contributions")
    start_time = perf_counter()

    # Now sum up each particle's contribution
    first_term = np.sum(first_term, axis=-2)
    # second_term = np.sum(second_term, axis=-2)

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Summing up detector field contributions took {duration:.4f} seconds")

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


if __name__ == "__main__":
    main()
