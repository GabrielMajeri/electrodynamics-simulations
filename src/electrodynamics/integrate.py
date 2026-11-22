from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from electrodynamics.beams import (
    GaussianBeamParameters,
    LaguerreGaussBeamParameters,
    LaserParameters,
    compute_directional_beam_fields,
    compute_oblique_beam_fields,
)
from electrodynamics.constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
)
from electrodynamics.fields import (
    compute_acceleration_of_charged_particle_in_em_field,
    compute_lab_time_lorentz_derivatives,
)
from electrodynamics.pulse import (
    GaussianPulseParameters,
    compute_pulse_envelope,
)

type AnyLaserParameters = (
    LaserParameters | GaussianBeamParameters | LaguerreGaussBeamParameters
)


@jdc.pytree_dataclass
class SnapshotData:
    "Recorded particle bunch snapshot at a simulation timestamp."

    time: float
    positions: jax.Array
    momenta: jax.Array


def parse_snapshot_schedule(
    snapshot_periods: str, laser_period: float, t_final_periods: float
) -> tuple[list[float], list[float]]:
    """Parses comma-separated snapshot timestamps (in laser periods `T_0`)
    and returns `(parsed_snapshot_periods, ordered_target_times_au)`.
    """
    parsed_snapshots = sorted(
        [float(s.strip()) for s in snapshot_periods.split(",") if s.strip()]
    )
    snapshot_times = [
        s * laser_period
        for s in parsed_snapshots
        if s * laser_period < t_final_periods * laser_period
    ]
    schedule_times = sorted(set(snapshot_times + [t_final_periods * laser_period]))
    return parsed_snapshots, schedule_times


@jdc.jit
def compute_electric_and_magnetic_fields(
    position: jax.Array,
    laser_parameters: jdc.Static[AnyLaserParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    incidence_angle: jdc.Static[float] = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Evaluates pulsed laser electric and magnetic fields at the given 4-positions."""
    modulation = jnp.expand_dims(
        compute_pulse_envelope(
            position, pulse_parameters, incidence_angle=incidence_angle
        ),
        axis=-1,
    )
    em_field = compute_oblique_beam_fields(
        laser_parameters, position, incidence_angle=incidence_angle
    )
    return modulation * em_field.electric_field, modulation * em_field.magnetic_field


@jdc.jit
def compute_intermediate_acceleration(
    position: jax.Array,
    momentum: jax.Array,
    laser_parameters: jdc.Static[AnyLaserParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    incidence_angle: jdc.Static[float] = 0.0,
) -> jax.Array:
    """Computes proper-time 4-acceleration du/dtau for a particle in the pulsed laser field."""
    electric_field, magnetic_field = compute_electric_and_magnetic_fields(
        position, laser_parameters, pulse_parameters, incidence_angle=incidence_angle
    )
    return compute_acceleration_of_charged_particle_in_em_field(
        momentum,
        electric_field,
        magnetic_field,
        charge_to_mass_ratio=ELECTRON_CHARGE / ELECTRON_MASS,
    )


def _rk4_step(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    deriv_fn: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
) -> tuple[jax.Array, jax.Array]:
    """Performs a single 4th-order Runge-Kutta step given a `(pos, mom)` derivative callable."""
    k1_pos, k1_mom = deriv_fn(previous_position, previous_momentum)
    k2_pos, k2_mom = deriv_fn(
        previous_position + (time_step / 2) * k1_pos,
        previous_momentum + (time_step / 2) * k1_mom,
    )
    k3_pos, k3_mom = deriv_fn(
        previous_position + (time_step / 2) * k2_pos,
        previous_momentum + (time_step / 2) * k2_mom,
    )
    k4_pos, k4_mom = deriv_fn(
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


@jdc.jit
def integration_step_rk4(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[AnyLaserParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    incidence_angle: jdc.Static[float] = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Updates particles' positions and momenta in proper time using 4th-order Runge-Kutta."""

    def deriv(pos: jax.Array, mom: jax.Array) -> tuple[jax.Array, jax.Array]:
        return mom, compute_intermediate_acceleration(
            pos,
            mom,
            laser_parameters,
            pulse_parameters,
            incidence_angle=incidence_angle,
        )

    return _rk4_step(previous_position, previous_momentum, time_step, deriv)


@jdc.jit
def compute_lab_time_derivatives(
    position: jax.Array,
    momentum: jax.Array,
    laser_parameters: jdc.Static[AnyLaserParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    incidence_angle: jdc.Static[float] = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Computes laboratory time derivatives `d(position)/dt` and `d(momentum)/dt`."""
    electric_field, magnetic_field = compute_electric_and_magnetic_fields(
        position, laser_parameters, pulse_parameters, incidence_angle=incidence_angle
    )
    return compute_lab_time_lorentz_derivatives(
        momentum,
        electric_field,
        magnetic_field,
        charge_to_mass_ratio=ELECTRON_CHARGE / ELECTRON_MASS,
    )


@jdc.jit
def integration_step_rk4_lab_time(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[AnyLaserParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    incidence_angle: jdc.Static[float] = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Updates particles' 4-positions and 4-momenta in laboratory time `dt` using 4th-order Runge-Kutta."""

    def deriv(pos: jax.Array, mom: jax.Array) -> tuple[jax.Array, jax.Array]:
        return compute_lab_time_derivatives(
            pos, mom, laser_parameters, pulse_parameters, incidence_angle
        )

    return _rk4_step(previous_position, previous_momentum, time_step, deriv)


@jdc.jit
def compute_two_pulse_fields(
    position: jax.Array,
    drive_laser: jdc.Static[AnyLaserParameters],
    drive_pulse: jdc.Static[GaussianPulseParameters],
    probe_laser: jdc.Static[AnyLaserParameters | None] = None,
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
    mod_probe = jnp.expand_dims(
        compute_pulse_envelope(
            position,
            probe_pulse,
            propagation_direction=probe_direction,
            reference_position=probe_reference_z,
        ),
        axis=-1,
    )

    E_total = E_drive + mod_probe * em_probe.electric_field
    B_total = B_drive + mod_probe * em_probe.magnetic_field
    return E_total, B_total


@jdc.jit
def compute_lab_time_derivatives_two_pulses(
    position: jax.Array,
    momentum: jax.Array,
    drive_laser: jdc.Static[AnyLaserParameters],
    drive_pulse: jdc.Static[GaussianPulseParameters],
    probe_laser: jdc.Static[AnyLaserParameters | None] = None,
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
    return compute_lab_time_lorentz_derivatives(
        momentum,
        electric_field,
        magnetic_field,
        charge_to_mass_ratio=ELECTRON_CHARGE / ELECTRON_MASS,
    )


@jdc.jit
def integration_step_rk4_lab_time_two_pulses(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: float,
    drive_laser: jdc.Static[AnyLaserParameters],
    drive_pulse: jdc.Static[GaussianPulseParameters],
    probe_laser: jdc.Static[AnyLaserParameters | None] = None,
    probe_pulse: jdc.Static[GaussianPulseParameters | None] = None,
    probe_direction: jdc.Static[int] = -1,
    probe_reference_z: jdc.Static[float] = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Updates particles' 4-positions and 4-momenta in laboratory time `dt` under two laser pulses."""

    def deriv(pos: jax.Array, mom: jax.Array) -> tuple[jax.Array, jax.Array]:
        return compute_lab_time_derivatives_two_pulses(
            pos,
            mom,
            drive_laser,
            drive_pulse,
            probe_laser,
            probe_pulse,
            probe_direction,
            probe_reference_z,
        )

    return _rk4_step(previous_position, previous_momentum, time_step, deriv)
