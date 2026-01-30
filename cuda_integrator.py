from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import cupy as cp
import matplotlib.pyplot as plt

### Physical constants (in atomic units)
# Speed of light
c = 137.036

# Angular frequency of laser radiation (we assume it's monochromatic)
omega_laser = 0.057

# Wavelength of laser radiation
lmbd = c * (2 * cp.pi / omega_laser)
# print("Lambda:", lmbd)

# Number of particles (electrons) to simulate
num_particles = 30


type Array = cp.ndarray[cp.float64]


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

    particle_indices = cp.arange(num_particles, dtype=cp.int32)

    # Determine the particles' center-of-motion
    centers = cp.array(
        (
            large_circle_radius * cp.cos(2 * cp.pi * particle_indices / num_particles),
            large_circle_radius * cp.sin(2 * cp.pi * particle_indices / num_particles),
            cp.zeros(num_particles),
        )
    ).T

    trajectories = cp.empty((num_particles, num_timestamps, 3))

    # Determine for how long the simulation/numerical integration will run
    # We'll use an integer multiple of the laser pulse's period
    num_periods = 40
    integration_duration = num_periods * (2 * cp.pi) / omega_laser
    timestamps = cp.linspace(0, integration_duration, num_timestamps, dtype=cp.float64)

    # We apply a decay to the trajectories to ensure the integral decays near the boundaries
    slope = 20
    cutoff = cp.exp(-1 / (timestamps / slope + 1e-10) ** 2) * cp.exp(
        -1 / (((integration_duration - timestamps) / slope + 1e-10) ** 2)
    )

    trajectory_radius = 0.75 * lmbd
    amplitude = trajectory_radius * cutoff

    # Compute instantaneous velocities and make sure we don't go faster than the speed of light
    # (this might happen if we set a trajectory with too large a circumference)
    assert cp.all(amplitude / (2 * cp.pi / omega_laser) < c), (
        "Particles move faster than speed of light"
    )

    num_wraps = 1
    phi_0 = particle_indices * num_wraps * (2 * cp.pi) / num_particles

    trajectories[:, :, 0] = centers[:, 0, cp.newaxis] + (
        amplitude
        * cp.cos(omega_laser * timestamps[cp.newaxis, :] - phi_0[:, cp.newaxis])
    )

    trajectories[:, :, 1] = centers[:, 1, cp.newaxis] + (
        amplitude
        * cp.sin(omega_laser * timestamps[cp.newaxis, :] - phi_0[:, cp.newaxis])
    )

    random_vertical_offsets = False
    if random_vertical_offsets:
        max_vertical_offset = 2 * trajectory_radius

        rng = cp.random.default_rng(15)
        vertical_offsets = rng.uniform(
            low=0, high=max_vertical_offset, size=num_particles
        )
        trajectories[:, :, 2] = vertical_offsets[:, cp.newaxis]
    else:
        trajectories[:, :, 2] = 0

    return CircularTrajectories(integration_duration, timestamps, centers, trajectories)


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
    detector_positions = cp.zeros(shape=(1024, 3), dtype=cp.float64)

    # Linearly displaced detector points
    detector_positions[:, 0] = cp.linspace(
        -30 * large_circle_radius,
        30 * large_circle_radius,
        detector_positions.shape[0],
    )

    # Radially displaced detector points
    # phi = np.linspace(0, 2 * np.pi, 1024)
    # detector_positions[:, 0] = 3.8 * large_circle_radius * np.cos(phi)
    # detector_positions[:, 1] = 3.8 * large_circle_radius * np.sin(phi)

    detector_positions[:, 2] = 1000 * lmbd

    # Current offset of particle from "center" of its motion
    particle_displacements = trajectories - centers[:, cp.newaxis, :]
    r_0s = particle_displacements

    # Compute detector displacement (in each particle frame of reference)
    x_0s = detector_positions[:, cp.newaxis, :] - centers[cp.newaxis, :, :]
    x_0s_norms = cp.linalg.norm(x_0s, axis=-1)

    n_0s = x_0s / x_0s_norms[:, :, cp.newaxis]

    dt = timestamps[1] - timestamps[0]

    frequency = omega_laser

    print("Computing (n_0, r_0(t)) dot products")

    start_time = perf_counter()

    n_0s_dot_r_0s = (n_0s[:, :, cp.newaxis, :] * r_0s[cp.newaxis, :, :, :]).sum(axis=-1)

    g = frequency * timestamps - frequency / c * n_0s_dot_r_0s

    exponent = 1j * g

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing exponents took {duration:.4g} seconds")

    start_time = perf_counter()

    n_0s_dot_v_0s = cp.gradient(n_0s_dot_r_0s, axis=-1)

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing velocities took {duration:.4g} seconds")

    start_time = perf_counter()

    oscillatory_kernel = cp.exp(exponent)
    integrand = oscillatory_kernel * n_0s_dot_v_0s

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing integrands took {duration:.4g} seconds")

    start_time = perf_counter()

    # Approximate integral using Riemann sum
    result = (1 / x_0s_norms) * dt * cp.sum(integrand, axis=-1)

    # Sum across particles
    result = cp.sum(result, axis=-1)

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Computing integrals using Riemann sums took {duration:.4g} seconds")

    final_time = perf_counter()
    total_duration = final_time - initial_time
    print(f"Execution took a total of {total_duration:.3g} seconds")

    print("Plotting results")

    plt.title("Integration results")

    plt.plot(
        detector_positions[:, 0].get(), cp.abs(result).get(), label="$|\\phi(x_0)|$"
    )
    # plt.plot(phi, np.angle(result), label="$\\arg \\phi(x_0)$")

    plt.xlabel("$x_0$")
    plt.ylabel("Scalar potential absolute value ($|\\phi|$)")

    # plt.xlabel("$\\phi$")
    # plt.ylabel("Scalar potential argument ($\\arg \\phi$)")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("plots/cuda_scalar_potential_vs_x_0.pdf")
    plt.close()
