import jax
import jax.numpy as jnp
import jax_dataclasses as jdc


@jdc.jit
def gaussian_envelope(
    phi: jax.Array, phi_0: jdc.Static[float], tau_0: jdc.Static[float]
) -> jax.Array:
    s = (phi - phi_0) / tau_0
    return jnp.exp(-jnp.square(s))


@jdc.jit
def gaussian_envelope_with_flat_peak(
    phi: jax.Array,
    phi_0: jdc.Static[float],
    tau_0: jdc.Static[float],
    peak_duration_periods: jdc.Static[int],
) -> jax.Array:
    s1 = (phi - phi_0) / tau_0
    s2 = (phi - peak_duration_periods * tau_0 - phi_0) / tau_0
    return jnp.where(
        phi < phi_0,
        jnp.exp(-jnp.square(s1)),
        jnp.where(
            phi < phi_0 + peak_duration_periods * tau_0, 1, jnp.exp(-jnp.square(s2))
        ),
    )


@jdc.jit
def cosine_squared_envelope(
    phi: jax.Array, phi_0: jdc.Static[float], tau_0: jdc.Static[float]
) -> jax.Array:
    """Cosine-squared temporal envelope:
    g(phi) = cos^2(pi * (phi - phi_0) / (2 * tau_0)) for |phi - phi_0| <= tau_0, else 0.
    """
    arg = (jnp.pi * (phi - phi_0)) / (2 * tau_0)
    in_pulse = jnp.abs(phi - phi_0) <= tau_0
    return jnp.where(in_pulse, jnp.square(jnp.cos(arg)), 0.0)


@jdc.pytree_dataclass
class GaussianPulseParameters:
    phi_0: float
    tau_0: float


@jdc.pytree_dataclass
class CosineSquaredPulseParameters(GaussianPulseParameters):
    "Parameters for a cosine-squared pulse envelope g(x - ct) = cos^2(pi * (x - ct) / (2 * c * tau_0))."


@jdc.pytree_dataclass
class PulseWithFlatPeakParameters(GaussianPulseParameters):
    peak_duration_periods: int


@jdc.jit
def compute_pulse_envelope(
    position: jax.Array,
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    propagation_direction: jdc.Static[int] = 1,
    reference_position: jdc.Static[float] = 0.0,
    c: jdc.Static[float] = 137.036,
) -> jax.Array:
    """Computes the scalar pulse envelope modulation at given 4-positions
    for a pulse moving in direction propagation_direction (+1 or -1) relative to reference_position.
    """
    ct, _, _, z = position.T
    phi = (ct - propagation_direction * (z - reference_position)) / c

    if isinstance(pulse_parameters, CosineSquaredPulseParameters):
        return cosine_squared_envelope(
            phi, pulse_parameters.phi_0, pulse_parameters.tau_0
        )
    elif isinstance(pulse_parameters, PulseWithFlatPeakParameters):
        return gaussian_envelope_with_flat_peak(
            phi,
            pulse_parameters.phi_0,
            pulse_parameters.tau_0,
            pulse_parameters.peak_duration_periods,
        )
    elif isinstance(pulse_parameters, GaussianPulseParameters):
        return gaussian_envelope(phi, pulse_parameters.phi_0, pulse_parameters.tau_0)
    else:
        return jnp.ones_like(phi)
