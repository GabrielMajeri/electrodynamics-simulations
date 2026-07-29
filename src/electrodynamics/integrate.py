import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from electrodynamics.beams import (
    LaguerreGaussBeamParameters,
    compute_laguerre_gauss_beam_fields,
)
from electrodynamics.constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    SPEED_OF_LIGHT as c,
)
from electrodynamics.fields import compute_acceleration_of_charged_particle_in_em_field
from electrodynamics.pulse import (
    PulseWithFlatPeakParameters,
    gaussian_envelope_with_flat_peak,
)


@jdc.jit
def compute_electric_and_magnetic_fields(
    position: jax.Array,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseWithFlatPeakParameters],
) -> tuple[jax.Array, jax.Array]:
    ct, _, _, z = position.T

    modulation = gaussian_envelope_with_flat_peak(
        (ct - z) / c,
        pulse_parameters.phi_0,
        pulse_parameters.tau_0,
        pulse_parameters.peak_duration_periods,
    )
    modulation = jnp.expand_dims(modulation, axis=-1)

    # electric_field, magnetic_field = compute_plane_wave_fields(
    #     laser_parameters, position
    # )
    electric_field, magnetic_field = compute_laguerre_gauss_beam_fields(
        laser_parameters, position
    )

    electric_field = modulation * electric_field
    magnetic_field = modulation * magnetic_field

    return electric_field, magnetic_field


@jdc.jit
def integration_step_euler(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseWithFlatPeakParameters],
) -> tuple[jax.Array, jax.Array]:
    electric_field, magnetic_field = compute_electric_and_magnetic_fields(
        previous_position,
        laser_parameters,
        pulse_parameters,
    )
    acceleration = compute_acceleration_of_charged_particle_in_em_field(
        previous_momentum,
        electric_field,
        magnetic_field,
        charge_to_mass_ratio=-1,
    )

    # TODO: add error checks

    new_momentum = previous_momentum + time_step * acceleration
    new_position = previous_position + time_step * new_momentum
    return new_position, new_momentum


@jdc.jit
def compute_intermediate_acceleration(
    position: jax.Array,
    momentum: jax.Array,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseWithFlatPeakParameters],
) -> jax.Array:
    electric_field, magnetic_field = compute_electric_and_magnetic_fields(
        position, laser_parameters, pulse_parameters
    )

    return compute_acceleration_of_charged_particle_in_em_field(
        momentum,
        electric_field,
        magnetic_field,
        charge_to_mass_ratio=ELECTRON_CHARGE / ELECTRON_MASS,
    )


@jdc.jit
def integration_step_rk4(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseWithFlatPeakParameters],
) -> tuple[jax.Array, jax.Array]:
    """Updates the particles' positions and momenta using
    a 4th order Runge-Kutta numerical integration scheme.
    """
    position_k_1 = time_step * previous_momentum
    momentum_k_1 = time_step * compute_intermediate_acceleration(
        previous_position,
        previous_momentum,
        laser_parameters,
        pulse_parameters,
    )

    position_k_2 = time_step * (previous_momentum + momentum_k_1 / 2)
    momentum_k_2 = time_step * compute_intermediate_acceleration(
        previous_position + position_k_1 / 2,
        previous_momentum + momentum_k_1 / 2,
        laser_parameters,
        pulse_parameters,
    )

    position_k_3 = time_step * (previous_momentum + momentum_k_2 / 2)
    momentum_k_3 = time_step * compute_intermediate_acceleration(
        previous_position + position_k_2 / 2,
        previous_momentum + momentum_k_2 / 2,
        laser_parameters,
        pulse_parameters,
    )

    position_k_4 = time_step * (previous_momentum + momentum_k_3)
    momentum_k_4 = time_step * compute_intermediate_acceleration(
        previous_position + position_k_3,
        previous_momentum + momentum_k_3,
        laser_parameters,
        pulse_parameters,
    )

    new_position = (
        previous_position
        + (position_k_1 + 2 * position_k_2 + 2 * position_k_3 + position_k_4) / 6
    )
    new_momentum = (
        previous_momentum
        + (momentum_k_1 + 2 * momentum_k_2 + 2 * momentum_k_3 + momentum_k_4) / 6
    )

    # TODO: add error checks

    return new_position, new_momentum
