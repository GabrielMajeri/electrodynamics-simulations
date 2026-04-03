import numpy as np

from .integrator import AccelerationFunction, symplectic_4th_order_step
from .typing import RealArray


def iterate_initial_conditions_hamiltonian_system(
    acceleration: AccelerationFunction,
    initial_positions: RealArray,
    initial_velocities: RealArray,
    integration_time: float = 100.0,
    time_step: float = 0.1,
) -> tuple[RealArray, RealArray, RealArray]:
    """Iterates an array of initial conditions for a fixed amount of time,
    returning the evaluation times, the positions along the trajectory,
    as well as the integrated velocities.

    Uses a 4-th order symplectic integration scheme.
    """
    times = np.arange(0.0, integration_time, time_step, dtype=np.float64)
    num_steps = len(times)

    num_particles = len(initial_positions)
    dimension = initial_positions.shape[-1]
    assert initial_velocities.shape[-1] == dimension, (
        "Initial velocities don't have same number of coordinates / components as initial positions"
    )

    # Allocate a buffer in which to store the generated trajectories
    positions = np.empty(
        (num_steps, num_particles, dimension),
        dtype=np.float64,
    )
    positions[0] = initial_positions

    velocities = np.empty(
        (num_steps, num_particles, dimension),
        dtype=np.float64,
    )
    velocities[0] = initial_velocities

    for index, _ in enumerate(times[1:], start=1):
        q = positions[index - 1]
        p = velocities[index - 1]

        q, p = symplectic_4th_order_step(q, p, time_step, acceleration)

        if np.isnan(q).any() or np.isnan(p).any():
            raise Exception(
                "Invalid value obtained in numerical integration code, try using a smaller time step or check your model"
            )

        positions[index] = q
        velocities[index] = p

    return times, positions, velocities
