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


@jdc.pytree_dataclass
class GaussianPulseParameters:
    phi_0: float
    tau_0: float


@jdc.pytree_dataclass
class PulseWithFlatPeakParameters(GaussianPulseParameters):
    peak_duration_periods: int
