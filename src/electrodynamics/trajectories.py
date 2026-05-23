from typing import NamedTuple

import numpy as np

from .constants import num_particles, omega_laser, c, lmbd
from .typing import RealArray


class CircularTrajectories(NamedTuple):
    integration_duration: float
    timestamps: RealArray
    centers: RealArray
    trajectories: RealArray
    velocities: RealArray


def simulate_circular_trajectories(
    # Radius of the large circle around which the particle's center are positioned.
    large_circle_radius,
    # Integration time delta (will affect trajectory sampling frequency)
    time_step=1,
) -> CircularTrajectories:
    """Code to numerically compute the particle's trajectories.

    They move around in a small circular trajectory around a center of motion,
    and these centers are radially distributed around the origin.
    """

    particle_indices = np.arange(num_particles, dtype=np.int32)

    # Determine the particles' center-of-motion
    centers = np.array(
        (
            large_circle_radius * np.cos(2 * np.pi * particle_indices / num_particles),
            large_circle_radius * np.sin(2 * np.pi * particle_indices / num_particles),
            np.zeros(num_particles),
        )
    ).T

    # Determine for how long the simulation/numerical integration will run
    # We'll use an integer multiple of the laser pulse's period
    # TODO: turn into a configurable parameter
    num_periods = 40
    integration_duration = num_periods * (2 * np.pi) / omega_laser

    print(
        f"Simulating trajectories from t = {0.0} to t = {integration_duration}, time step = {time_step}"
    )

    num_timestamps = int(integration_duration / time_step) + 1
    trajectories = np.empty((num_particles, num_timestamps, 3))

    timestamps = np.linspace(0, integration_duration, num_timestamps)

    # We apply a decay to the trajectories to ensure the integral decays near the boundaries
    # TODO: turn into a configurable parameter
    slope = 20
    cutoff = np.exp(-1 / (timestamps / slope + 1e-10) ** 2) * np.exp(
        -1 / (((integration_duration - timestamps) / slope + 1e-10) ** 2)
    )

    trajectory_radius = (0.1 / (2 * np.pi)) * lmbd
    amplitude = trajectory_radius * cutoff

    # Compute instantaneous velocities and make sure we don't go faster than the speed of light
    # (this might happen if we set a trajectory with too large a circumference)
    assert np.all(amplitude / (2 * np.pi / omega_laser) < c), (
        "Particles move faster than speed of light"
    )

    # TODO: turn into a configurable parameter
    num_wraps = 1
    phi_0 = particle_indices * num_wraps * (2 * np.pi) / num_particles

    trajectories[:, :, 0] = centers[:, 0, np.newaxis] + (
        amplitude
        * np.cos(omega_laser * timestamps[np.newaxis, :] - phi_0[:, np.newaxis])
    )

    trajectories[:, :, 1] = centers[:, 1, np.newaxis] + (
        amplitude
        * np.sin(omega_laser * timestamps[np.newaxis, :] - phi_0[:, np.newaxis])
    )

    velocities = np.empty((num_particles, num_timestamps, 3))
    velocities[:, :, 0] = (
        -amplitude
        * omega_laser
        * np.sin(omega_laser * timestamps[np.newaxis, :] - phi_0[:, np.newaxis])
    )
    velocities[:, :, 1] = (
        amplitude
        * omega_laser
        * np.cos(omega_laser * timestamps[np.newaxis, :] - phi_0[:, np.newaxis])
    )
    velocities[:, :, 2] = 0

    # TODO: turn into a configurable parameter
    random_vertical_offsets = False
    if random_vertical_offsets:
        max_vertical_offset = 2 * trajectory_radius

        rng = np.random.default_rng(15)
        vertical_offsets = rng.uniform(
            low=0, high=max_vertical_offset, size=num_particles
        )
        trajectories[:, :, 2] = vertical_offsets[:, np.newaxis]
    else:
        trajectories[:, :, 2] = 0

    return CircularTrajectories(
        integration_duration, timestamps, centers, trajectories, velocities
    )
