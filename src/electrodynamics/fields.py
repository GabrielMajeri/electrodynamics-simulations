import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from electrodynamics.constants import SPEED_OF_LIGHT

c = SPEED_OF_LIGHT


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
def compute_scattered_electric_and_magnetic_fields(
    current_time: float,
    frequency: float | jax.Array,
    position: jax.Array,
    momentum: jax.Array,
    initial_position: jax.Array,
    detector_position: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Computes the scattered electric/magnetic field from a moving charge,
    using the formulas for the Liénard-Wiechert potentials.
    """
    particle_position = position[:, 1:4]
    particle_velocity = momentum[:, 1:4]
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

    # exp(i * omega * (t + R(x_0, t)/c))
    oscillatory_kernel = jnp.exp(
        1j * frequency * (current_time + displacement_norm / c)
    )

    # \beta = v/c
    beta = particle_velocity / c

    # ===== Electric field terms =====
    # Common term: n(x_0, t) \times (n(x_0, t) \times \beta(t))
    electric_field_common_term = jnp.cross(
        view_direction, jnp.cross(view_direction, beta)
    )

    # O(1/|R|) term
    # - ((i * omega) / c) * (common term) / |R(x_0, t)|
    electric_field_first_term = -((1j * frequency) / c) * (
        electric_field_common_term / displacement_norm
    )

    displacement_norm_squared = displacement_norm * displacement_norm

    # O(1/|R|^2) term
    # [(common term) + n(x_0, t) * (1 + dot(n(x_0, t), \beta(t)))] / |R(x_0, t)|^2
    electric_field_second_term = (
        electric_field_common_term
        + view_direction
        * jnp.expand_dims(1 + jnp.linalg.vecdot(view_direction, beta), axis=-1)
    ) / displacement_norm_squared

    n_cross_beta = jnp.cross(view_direction, beta)

    # ===== Magnetic field terms =====
    # O(1/|R|) term
    magnetic_field_first_term = ((1j * frequency) / c) * (
        n_cross_beta / displacement_norm
    )

    # O(1/|R|^2) term
    magnetic_field_second_term = n_cross_beta / displacement_norm_squared

    # Add up the components
    electric_field = oscillatory_kernel * (
        electric_field_first_term + electric_field_second_term
    )
    magnetic_field = oscillatory_kernel * (
        magnetic_field_first_term + magnetic_field_second_term
    )

    return electric_field, magnetic_field
