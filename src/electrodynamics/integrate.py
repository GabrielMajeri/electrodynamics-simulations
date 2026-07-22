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
    time: float,
    position: jax.Array,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseWithFlatPeakParameters],
) -> tuple[jax.Array, jax.Array]:
    _, _, _, z = position.T

    modulation = gaussian_envelope_with_flat_peak(
        time - z / c,
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
def compute_next_momentum_euler(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseWithFlatPeakParameters],
) -> jax.Array:
    tc, _, _, _ = previous_position.T
    laboratory_time = tc / c

    electric_field, magnetic_field = compute_electric_and_magnetic_fields(
        laboratory_time,
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

    return previous_momentum + time_step * acceleration


@jdc.jit
def compute_intermediate_acceleration(
    time: float,
    position: jax.Array,
    momentum: jax.Array,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseWithFlatPeakParameters],
) -> jax.Array:
    electric_field, magnetic_field = compute_electric_and_magnetic_fields(
        time, position, laser_parameters, pulse_parameters
    )

    return compute_acceleration_of_charged_particle_in_em_field(
        momentum,
        electric_field,
        magnetic_field,
        charge_to_mass_ratio=ELECTRON_CHARGE / ELECTRON_MASS,
    )


@jdc.jit
def compute_next_momentum_rk4(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[PulseWithFlatPeakParameters],
) -> jax.Array:
    "Computes momentum at next time step using the 4th-order Runge-Kutta method."

    tc, _, _, _ = previous_position.T
    laboratory_time = tc / c

    # Runge-Kutta 4th order
    k_1 = compute_intermediate_acceleration(
        laboratory_time,
        previous_position,
        previous_momentum,
        laser_parameters,
        pulse_parameters,
    )

    k_2 = compute_intermediate_acceleration(
        laboratory_time + time_step / 2,
        previous_position,
        previous_momentum + time_step / 2 * k_1,
        laser_parameters,
        pulse_parameters,
    )

    k_3 = compute_intermediate_acceleration(
        laboratory_time + time_step / 2,
        previous_position,
        previous_momentum + time_step / 2 * k_2,
        laser_parameters,
        pulse_parameters,
    )

    k_4 = compute_intermediate_acceleration(
        laboratory_time + time_step,
        previous_position,
        previous_momentum + time_step * k_3,
        laser_parameters,
        pulse_parameters,
    )

    acceleration = (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    # TODO: implement sanity checks
    # if check_for_errors:
    #     check_integration_results(previous_momentum, acceleration)

    return previous_momentum + time_step * acceleration
