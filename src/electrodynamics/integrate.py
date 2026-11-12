import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from electrodynamics.beams import (
    GaussianBeamParameters,
    LaguerreGaussBeamParameters,
    LaserParameters,
    compute_directional_beam_fields,
    compute_laguerre_gauss_beam_fields,
)
from electrodynamics.constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    SPEED_OF_LIGHT,
)
from electrodynamics.fields import compute_acceleration_of_charged_particle_in_em_field
from electrodynamics.pulse import (
    CosineSquaredPulseParameters,
    GaussianPulseParameters,
    PulseWithFlatPeakParameters,
    compute_pulse_envelope,
    cosine_squared_envelope,
    gaussian_envelope,
    gaussian_envelope_with_flat_peak,
)


@jdc.jit
def compute_electric_and_magnetic_fields(
    position: jax.Array,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
) -> tuple[jax.Array, jax.Array]:
    ct, _, _, z = position.T

    phi = (ct - z) / SPEED_OF_LIGHT
    if isinstance(pulse_parameters, CosineSquaredPulseParameters):
        modulation = cosine_squared_envelope(
            phi, pulse_parameters.phi_0, pulse_parameters.tau_0
        )
    elif isinstance(pulse_parameters, PulseWithFlatPeakParameters):
        modulation = gaussian_envelope_with_flat_peak(
            phi,
            pulse_parameters.phi_0,
            pulse_parameters.tau_0,
            pulse_parameters.peak_duration_periods,
        )
    elif isinstance(pulse_parameters, GaussianPulseParameters):
        modulation = gaussian_envelope(
            phi, pulse_parameters.phi_0, pulse_parameters.tau_0
        )
    else:
        modulation = jnp.ones_like(phi)

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
        charge_to_mass_ratio=ELECTRON_CHARGE / ELECTRON_MASS,
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


@jdc.jit
def compute_lab_time_derivatives(
    position: jax.Array,
    momentum: jax.Array,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
) -> tuple[jax.Array, jax.Array]:
    """Computes the laboratory time derivatives d(position)/dt and d(momentum)/dt:
    d(position)/dt = (c, v_x, v_y, v_z)
    d(momentum)/dt = (du^0/dt, du^1/dt, du^2/dt, du^3/dt)
    where v = c * u / u^0 and du/dt = (q/m)*(E + v x B).
    """
    electric_field, magnetic_field = compute_electric_and_magnetic_fields(
        position, laser_parameters, pulse_parameters
    )

    u0 = momentum[..., 0:1]
    u = momentum[..., 1:4]

    gamma = u0 / SPEED_OF_LIGHT
    v = u / gamma

    q_over_m = ELECTRON_CHARGE / ELECTRON_MASS
    v_cross_B = jnp.cross(v, magnetic_field)

    du = q_over_m * (electric_field + v_cross_B)
    du0 = (q_over_m / SPEED_OF_LIGHT) * jnp.sum(
        v * electric_field, axis=-1, keepdims=True
    )

    d_momentum = jnp.concatenate([du0, du], axis=-1)
    d_position = jnp.concatenate([jnp.full_like(u0, SPEED_OF_LIGHT), v], axis=-1)

    return d_position, d_momentum


@jdc.jit
def integration_step_rk4_lab_time(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[LaguerreGaussBeamParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
) -> tuple[jax.Array, jax.Array]:
    """Updates particles' 4-positions and 4-momenta in laboratory time dt using 4th-order Runge-Kutta."""
    k1_pos, k1_mom = compute_lab_time_derivatives(
        previous_position, previous_momentum, laser_parameters, pulse_parameters
    )

    k2_pos, k2_mom = compute_lab_time_derivatives(
        previous_position + (time_step / 2) * k1_pos,
        previous_momentum + (time_step / 2) * k1_mom,
        laser_parameters,
        pulse_parameters,
    )

    k3_pos, k3_mom = compute_lab_time_derivatives(
        previous_position + (time_step / 2) * k2_pos,
        previous_momentum + (time_step / 2) * k2_mom,
        laser_parameters,
        pulse_parameters,
    )

    k4_pos, k4_mom = compute_lab_time_derivatives(
        previous_position + time_step * k3_pos,
        previous_momentum + time_step * k3_mom,
        laser_parameters,
        pulse_parameters,
    )

    new_position = previous_position + (time_step / 6) * (
        k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos
    )
    new_momentum = previous_momentum + (time_step / 6) * (
        k1_mom + 2 * k2_mom + 2 * k3_mom + k4_mom
    )

    return new_position, new_momentum


@jdc.jit
def compute_two_pulse_fields(
    position: jax.Array,
    drive_laser: jdc.Static[LaguerreGaussBeamParameters],
    drive_pulse: jdc.Static[GaussianPulseParameters],
    probe_laser: jdc.Static[
        LaserParameters | GaussianBeamParameters | LaguerreGaussBeamParameters | None
    ] = None,
    probe_pulse: jdc.Static[GaussianPulseParameters | None] = None,
    probe_direction: jdc.Static[int] = -1,
    probe_reference_z: jdc.Static[float] = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Computes total electric and magnetic fields from a primary drive pulse (+z)
    and an optional counter-propagating probe pulse.
    """
    E_drive, B_drive = compute_electric_and_magnetic_fields(
        position, drive_laser, drive_pulse
    )

    if probe_laser is None or probe_pulse is None:
        return E_drive, B_drive

    em_probe = compute_directional_beam_fields(
        probe_laser,
        position,
        propagation_direction=probe_direction,
        focal_position=probe_reference_z,
    )
    mod_probe = compute_pulse_envelope(
        position,
        probe_pulse,
        propagation_direction=probe_direction,
        reference_position=probe_reference_z,
    )
    mod_probe = jnp.expand_dims(mod_probe, axis=-1)

    E_total = E_drive + mod_probe * em_probe.electric_field
    B_total = B_drive + mod_probe * em_probe.magnetic_field
    return E_total, B_total


@jdc.jit
def compute_lab_time_derivatives_two_pulses(
    position: jax.Array,
    momentum: jax.Array,
    drive_laser: jdc.Static[LaguerreGaussBeamParameters],
    drive_pulse: jdc.Static[GaussianPulseParameters],
    probe_laser: jdc.Static[
        LaserParameters | GaussianBeamParameters | LaguerreGaussBeamParameters | None
    ] = None,
    probe_pulse: jdc.Static[GaussianPulseParameters | None] = None,
    probe_direction: jdc.Static[int] = -1,
    probe_reference_z: jdc.Static[float] = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Computes lab-time derivatives under primary drive pulse and counter-propagating probe pulse."""
    electric_field, magnetic_field = compute_two_pulse_fields(
        position,
        drive_laser,
        drive_pulse,
        probe_laser,
        probe_pulse,
        probe_direction,
        probe_reference_z,
    )

    u0 = momentum[..., 0:1]
    u = momentum[..., 1:4]

    gamma = u0 / SPEED_OF_LIGHT
    v = u / gamma

    q_over_m = ELECTRON_CHARGE / ELECTRON_MASS
    v_cross_B = jnp.cross(v, magnetic_field)

    du = q_over_m * (electric_field + v_cross_B)
    du0 = (q_over_m / SPEED_OF_LIGHT) * jnp.sum(
        v * electric_field, axis=-1, keepdims=True
    )

    d_momentum = jnp.concatenate([du0, du], axis=-1)
    d_position = jnp.concatenate([jnp.full_like(u0, SPEED_OF_LIGHT), v], axis=-1)

    return d_position, d_momentum


@jdc.jit
def integration_step_rk4_lab_time_two_pulses(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    drive_laser: jdc.Static[LaguerreGaussBeamParameters],
    drive_pulse: jdc.Static[GaussianPulseParameters],
    probe_laser: jdc.Static[
        LaserParameters | GaussianBeamParameters | LaguerreGaussBeamParameters | None
    ] = None,
    probe_pulse: jdc.Static[GaussianPulseParameters | None] = None,
    probe_direction: jdc.Static[int] = -1,
    probe_reference_z: jdc.Static[float] = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Updates particles' 4-positions and 4-momenta in laboratory time dt under two laser pulses."""

    def deriv(p, m):
        return compute_lab_time_derivatives_two_pulses(
            p,
            m,
            drive_laser,
            drive_pulse,
            probe_laser,
            probe_pulse,
            probe_direction,
            probe_reference_z,
        )

    k1_pos, k1_mom = deriv(previous_position, previous_momentum)
    k2_pos, k2_mom = deriv(
        previous_position + (time_step / 2) * k1_pos,
        previous_momentum + (time_step / 2) * k1_mom,
    )
    k3_pos, k3_mom = deriv(
        previous_position + (time_step / 2) * k2_pos,
        previous_momentum + (time_step / 2) * k2_mom,
    )
    k4_pos, k4_mom = deriv(
        previous_position + time_step * k3_pos,
        previous_momentum + time_step * k3_mom,
    )

    new_position = previous_position + (time_step / 6) * (
        k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos
    )
    new_momentum = previous_momentum + (time_step / 6) * (
        k1_mom + 2 * k2_mom + 2 * k3_mom + k4_mom
    )
    return new_position, new_momentum
