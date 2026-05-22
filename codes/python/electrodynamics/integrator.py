from typing import Callable

from .typing import RealArray

type AccelerationFunction = Callable[[RealArray, RealArray], RealArray]


def symplectic_euler_step(
    q: RealArray, p: RealArray, dt: float, acceleration: AccelerationFunction
) -> tuple[RealArray, RealArray]:
    """Symplectic Euler integration scheme."""

    # p(t + dt) = p + dt * a(t)
    p = p + dt * acceleration(q, p)

    # q(t + dt) = q(t) + dt * p(t + dt)
    q = q + dt * p

    return q, p


# Magic number which shows up in the coefficients of
# the 4-th order symplectic integration scheme,
# denoted by `x` in Forest's and Ruth's paper.
INTEGRATOR_CONSTANT = (2 ** (1 / 3) + 2 ** (-1 / 3) - 1) / 6


def symplectic_4th_order_step(
    q: RealArray, p: RealArray, dt: float, acceleration: AccelerationFunction
) -> tuple[RealArray, RealArray]:
    """Symplectic 4-th order integration scheme, as described by Forest and Ruth
    in https://www.sciencedirect.com/science/article/abs/pii/016727899090019L
    """

    # First quarter-step
    p = p + dt * (INTEGRATOR_CONSTANT + 0.5) * acceleration(q, p)
    q = q + dt * (2 * INTEGRATOR_CONSTANT + 1) * p

    # Second quarter-step
    p = p + dt * (-INTEGRATOR_CONSTANT) * acceleration(q, p)
    q = q + dt * (-4 * INTEGRATOR_CONSTANT - 1) * p

    # Third quarter-step
    p = p + dt * (-INTEGRATOR_CONSTANT) * acceleration(q, p)
    q = q + dt * (2 * INTEGRATOR_CONSTANT + 1) * p

    # Fourth quarter-step (final)
    p = p + dt * (INTEGRATOR_CONSTANT + 0.5) * acceleration(q, p)
    # q = q + 0

    return q, p
