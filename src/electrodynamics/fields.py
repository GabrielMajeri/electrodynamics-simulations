from math import atan, atan2, cos, pi, sin, sqrt

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from electrodynamics.constants import SPEED_OF_LIGHT as c


def compute_relativistic_reflection_parameters(
    gamma: float, incidence_angle: float = 0.0
) -> tuple[float, float, float, float]:
    """Computes the exact relativistic reflection parameters for a laser beam
    impinging at angle `incidence_angle` (relative to the mirror normal) on a
    relativistic flying mirror moving at speed `v = beta * c` towards the incoming beam,
    following Chen et al., Scientific Reports 12:12524 (2022):

    - In the flying mirror rest frame:
        tan(theta'_in) = sin(theta_in) / (gamma * (cos(theta_in) + beta))
        lambda'_0 = lambda_0 / (gamma * (1 + beta * cos(theta_in)))
    - In the laboratory frame:
        tan(theta_re) = sin(theta'_in) / (gamma * (cos(theta'_in) + beta))
        lambda_1 = lambda'_0 / (gamma * (1 + beta * cos(theta'_in)))
        doppler_factor = omega_1 / omega_0 = lambda_0 / lambda_1
    - Profile tilt angle phi between reflected linear momentum and electric field profile:
        sin(phi) / sin(phi + theta_in + theta_re) = beta * cos(theta_in) / cos(theta_re)

    Returns:
        (theta_in_mirror_frame, theta_re, doppler_factor, profile_tilt_angle)
    """
    if gamma < 1.0:
        raise ValueError(f"gamma must satisfy gamma >= 1, got {gamma}")

    beta = sqrt(1.0 - 1.0 / (gamma**2))

    theta_in_prime = atan2(sin(incidence_angle), gamma * (cos(incidence_angle) + beta))
    theta_re = atan2(sin(theta_in_prime), gamma * (cos(theta_in_prime) + beta))

    doppler_factor = (
        gamma
        * (1.0 + beta * cos(incidence_angle))
        * gamma
        * (1.0 + beta * cos(theta_in_prime))
    )

    # Profile tilt angle phi from
    #   sin(phi)/sin(phi + theta_in + theta_re) = beta * cos(theta_in) / cos(theta_re)
    ratio = (beta * cos(incidence_angle)) / cos(theta_re)
    total_angle = incidence_angle + theta_re
    denom = 1.0 - ratio * cos(total_angle)
    profile_tilt_angle = (
        atan((ratio * sin(total_angle)) / denom) if abs(denom) > 1e-12 else 0.0
    )

    return theta_in_prime, theta_re, doppler_factor, profile_tilt_angle


def compute_doppler_shift_factor(
    gamma: float, particle_mass: float, particle_charge: float, a_0: float
) -> float:
    """Computes the Doppler shift factor, the multiplier for the frequency
    of the scattered radiation emitted from a particle moving at relativistic speeds.
    """
    mc = particle_mass * c

    mc_gamma = mc * gamma
    mc_sqrt_gamma = mc * sqrt(gamma**2 - 1)

    return (mc_gamma + mc_sqrt_gamma) / (
        mc_gamma
        - mc_sqrt_gamma
        + (a_0**2 * particle_charge**2) / (mc_gamma + mc_sqrt_gamma)
    )


@jdc.jit
def compute_acceleration_of_charged_particle_in_em_field(
    momentum: jax.Array,
    electric_field: jax.Array,
    magnetic_field: jax.Array,
    charge_to_mass_ratio: jdc.Static[float],
) -> jax.Array:
    "Computes the acceleration felt by a charged particle in the presence of an electromagnetic field."

    u0, u1, u2, u3 = momentum.T
    E_x, E_y, E_z = electric_field.T
    B_x, B_y, B_z = magnetic_field.T

    du0 = u1 * E_x / c + u2 * E_y / c + u3 * E_z / c
    du1 = u0 * E_x / c + u2 * B_z - u3 * B_y
    du2 = u0 * E_y / c - u1 * B_z + u3 * B_x
    du3 = u0 * E_z / c + u1 * B_y - u2 * B_x

    return charge_to_mass_ratio * jnp.array((du0, du1, du2, du3)).T


@jdc.jit
def compute_lab_time_lorentz_derivatives(
    momentum: jax.Array,
    electric_field: jax.Array,
    magnetic_field: jax.Array,
    charge_to_mass_ratio: jdc.Static[float],
) -> tuple[jax.Array, jax.Array]:
    """Computes laboratory time derivatives `d(position)/dt` and `d(momentum)/dt`:
      `d(position)/dt = (c, v_x, v_y, v_z)`,
      `d(momentum)/dt = (du^0/dt, du^1/dt, du^2/dt, du^3/dt)`,
    where `v = c * u / u^0` and `du/dt = (q/m) * (E + v x B)`.
    """
    u0 = momentum[..., 0:1]
    u = momentum[..., 1:4]

    gamma = u0 / c
    v = u / gamma

    v_cross_B = jnp.cross(v, magnetic_field)

    du = charge_to_mass_ratio * (electric_field + v_cross_B)
    du0 = (charge_to_mass_ratio / c) * jnp.sum(
        v * electric_field, axis=-1, keepdims=True
    )

    d_momentum = jnp.concatenate([du0, du], axis=-1)
    d_position = jnp.concatenate([jnp.full_like(u0, c), v], axis=-1)

    return d_position, d_momentum


@jdc.jit
def compute_scattered_electric_and_magnetic_fields(
    frequency: float | jax.Array,
    position: jax.Array,
    momentum: jax.Array,
    initial_position: jax.Array,
    detector_position: jax.Array,
    include_second_order_terms: jdc.Static[bool] = False,
    is_lab_time: jdc.Static[bool] = False,
    initial_momentum: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Computes the scattered electric/magnetic field from a moving charge,
    using the formulas for the Liénard-Wiechert potentials.

    When `initial_momentum` is provided, the unperturbed relativistic drift velocity
    beta_0 = u_init / u^0_init is subtracted from beta(t) so that uniform coasting
    outside the laser pulse produces zero spurious boundary-window radiation.
    """
    laboratory_time = jnp.expand_dims(position[:, 0], axis=-1) / c
    particle_position = position[:, 1:4]
    # Note: momentum is 4-momentum u^\mu = (\gamma c, \gamma v) (with particle mass = 1 in atomic units).
    # Since trajectory integration is performed with respect to proper time \tau (d\tau = dt / \gamma),
    # the integration factor dt = \gamma d\tau absorbs \gamma so that \beta_code = particle_velocity / c = \gamma \beta.
    # When is_lab_time is True, trajectory integration is in laboratory time t (dt), so \beta = v / c = (u / \gamma) / c.
    particle_velocity = momentum[:, 1:4]
    gamma_factor = jnp.expand_dims(momentum[:, 0] / c, axis=-1)
    initial_position = initial_position[:, 1:4]

    # r_0(t) = r(t) - R_0
    particle_displacement = particle_position - initial_position

    # x_0(t) = x - R_0
    detector_displacement = detector_position - initial_position

    # R(x_0, t) = x_0 - r_0(t) = (x - R_0) - (r(t) - R_0) = x - r(t)
    displacement = detector_displacement - particle_displacement
    displacement_norm = jnp.expand_dims(
        jnp.linalg.vector_norm(displacement, axis=-1), axis=-1
    )

    # n(x_0, t) = R(x_0, t)/|R(x_0, t)|
    view_direction = displacement / displacement_norm

    # Exact cancellation-free evaluation of t + |x - r(t)|/c:
    # |x - r| = |x| + (-2 (x . r) + |r|^2) / (|x - r| + |x|)
    detector_norm = jnp.expand_dims(
        jnp.linalg.vector_norm(detector_position, axis=-1), axis=-1
    )
    r_squared = jnp.expand_dims(
        jnp.sum(particle_position * particle_position, axis=-1), axis=-1
    )
    x_dot_r = jnp.expand_dims(
        jnp.sum(detector_position * particle_position, axis=-1), axis=-1
    )
    delta_r = (-2.0 * x_dot_r + r_squared) / (displacement_norm + detector_norm + 1e-30)

    # exp(i * omega * (t + R(x_0, t)/c))
    oscillatory_kernel = jnp.exp(
        1j * frequency * (laboratory_time + delta_r / c)
    ) * jnp.exp(1j * frequency * (detector_norm / c))

    if is_lab_time:
        beta = (particle_velocity / gamma_factor) / c
        if initial_momentum is not None:
            init_gamma = jnp.expand_dims(initial_momentum[:, 0] / c, axis=-1)
            beta_0 = (initial_momentum[:, 1:4] / init_gamma) / c
            beta = beta - beta_0
    else:
        # \gamma \beta = u / c (scaled by dt/d\tau = \gamma for proper time integration)
        beta = particle_velocity / c
        if initial_momentum is not None:
            init_gamma = jnp.expand_dims(initial_momentum[:, 0] / c, axis=-1)
            beta_0 = (initial_momentum[:, 1:4] / init_gamma) / c
            beta = beta - gamma_factor * beta_0

    # ===== Electric field terms =====
    # Common term: n(x_0, t) \times (n(x_0, t) \times \beta(t))
    electric_field_common_term = jnp.cross(
        view_direction, jnp.cross(view_direction, beta)
    )

    # Electric field O(1/|R|) term
    # - ((i * omega) / c) * (common term) / |R(x_0, t)|
    electric_field_first_term = -((1j * frequency) / c) * (
        electric_field_common_term / displacement_norm
    )

    n_cross_beta = jnp.cross(view_direction, beta)

    # Magnetic field O(1/|R|) term
    magnetic_field_first_term = ((1j * frequency) / c) * (
        n_cross_beta / displacement_norm
    )

    # O(1/|R|^2) terms
    if include_second_order_terms:
        displacement_norm_squared = displacement_norm * displacement_norm

        # [(common term) + n(x_0, t) * (\gamma + dot(n(x_0, t), \gamma \beta(t)))] / |R(x_0, t)|^2
        electric_field_second_term = (
            electric_field_common_term
            + view_direction
            * (
                (1.0 if is_lab_time else gamma_factor)
                + jnp.expand_dims(jnp.linalg.vecdot(view_direction, beta), axis=-1)
            )
        ) / displacement_norm_squared

        # n(x_0, t) \times \beta(t) / |R(x_0, t)|^2
        magnetic_field_second_term = n_cross_beta / displacement_norm_squared

        # Add up the components
        electric_field_terms = electric_field_first_term + electric_field_second_term
        magnetic_field_terms = magnetic_field_first_term + magnetic_field_second_term

    else:
        # Use only the first-order terms
        electric_field_terms = electric_field_first_term
        magnetic_field_terms = magnetic_field_first_term

    electric_field = oscillatory_kernel * electric_field_terms
    magnetic_field = oscillatory_kernel * magnetic_field_terms

    return electric_field / (2 * pi), magnetic_field / (2 * pi)


@jdc.jit
def compute_coherent_detector_and_spectrum_fields(
    positions: jax.Array,
    momenta: jax.Array,
    origin_positions: jax.Array,
    detector_positions: jax.Array,
    radiation_frequency: float | jax.Array,
    spectrum_position: jax.Array,
    frequencies: jax.Array,
    is_lab_time: jdc.Static[bool] = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluates scattered EM fields from all particles, summing up the total coherently.

    Returns the total scattered electric and magnetic fields at the detector positions,
    and the total scattered electric field at the spectrum position,
    for each frequency in the `frequencies` array.
    """

    def compute_pixel(dp: jax.Array) -> tuple[jax.Array, jax.Array]:
        E, B = compute_scattered_electric_and_magnetic_fields(
            radiation_frequency,
            positions,
            momenta,
            origin_positions,
            dp,
            is_lab_time=is_lab_time,
        )
        return jnp.sum(E, axis=0), jnp.sum(B, axis=0)

    dE_det, dB_det = jax.vmap(compute_pixel)(detector_positions)

    def compute_spec(w: jax.Array) -> jax.Array:
        e, _ = compute_scattered_electric_and_magnetic_fields(
            w,
            positions,
            momenta,
            origin_positions,
            spectrum_position,
            is_lab_time=is_lab_time,
        )
        return jnp.sum(e, axis=0)

    dE_spec = jax.vmap(compute_spec)(frequencies)
    return dE_det, dB_det, dE_spec
