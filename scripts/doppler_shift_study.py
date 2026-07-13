from math import pi
import multiprocessing
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
import typer

from electrodynamics.constants import SPEED_OF_LIGHT
from electrodynamics.initial_conditions import generate_initial_positions_on_disk
from electrodynamics.plotting import plot_angular_momentum_distribution

c = SPEED_OF_LIGHT


@jdc.pytree_dataclass
class Polarization:
    x: complex
    y: complex

    def __init__(self, x: jdc.Static[complex], y: jdc.Static[complex]) -> None:
        norm = abs(x) + abs(y)
        if abs(norm - 1) > 1e-10:
            raise Exception("Polarization should have unit norm")

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


@jdc.pytree_dataclass
class LaserParameters:
    frequency: float
    wavelength: float
    amplitude: float
    polarization: Polarization

    def __init__(
        self,
        frequency: jdc.Static[complex],
        amplitude: jdc.Static[complex],
        polarization: jdc.Static[Polarization],
    ) -> None:
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "wavelength", (2 * pi * c) / frequency)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "polarization", polarization)


@jdc.pytree_dataclass
class IntegrationResult:
    final_positions: jax.Array
    final_momenta: jax.Array


@jdc.jit
def plane_wave_fields(
    parameters: jdc.Static[LaserParameters], position: jax.Array
) -> tuple[jax.Array, jax.Array]:
    tc, _, _, z = position.T
    t = tc / c

    # k
    wavenumber = (2 * pi) / parameters.wavelength

    magnitude = parameters.amplitude
    phase = jnp.exp(1j * (parameters.frequency * t - wavenumber * z))

    phasor = magnitude * phase

    polarization = parameters.polarization

    E_x = jnp.real(polarization.x * phasor)
    E_y = jnp.real(polarization.y * phasor)
    E_z = jnp.zeros_like(E_x)

    E = jnp.vstack((E_x, E_y, E_z)).T

    B_x = -E_y / c
    B_y = E_x / c
    B_z = jnp.zeros_like(B_x)

    B = jnp.vstack((B_x, B_y, B_z)).T

    return E, B


@jdc.jit
def cutoff(
    phi: jax.Array, phi_0: jdc.Static[float], tau_0: jdc.Static[float]
) -> jax.Array:
    t = (phi - phi_0) / tau_0
    return jnp.exp(-(t * t))


@jdc.pytree_dataclass
class PulseParameters:
    phi_0: float
    tau_0: float


@jdc.jit
def compute_acceleration(
    previous_momentum: jax.Array,
    electric_field: jax.Array,
    magnetic_field: jax.Array,
    charge_to_mass_ratio: jdc.Static[float],
) -> jax.Array:
    "Computes the acceleration felt by a charged particle in an electric field."

    u0, u1, u2, u3 = previous_momentum.T
    E_x, E_y, E_z = electric_field.T
    B_x, B_y, B_z = magnetic_field.T

    du0 = u1 * E_x / c + u2 * E_y / c + u3 * E_z / c
    du1 = u0 * E_x / c + u2 * B_z - u3 * B_y
    du2 = u0 * E_y / c - u1 * B_z + u3 * B_x
    du3 = u0 * E_z / c + u1 * B_y - u2 * B_x

    return charge_to_mass_ratio * jnp.array((du0, du1, du2, du3)).T


@jdc.jit
def compute_new_momentum(
    previous_position: jax.Array,
    previous_momentum: jax.Array,
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaserParameters],
    pulse_parameters: jdc.Static[PulseParameters],
) -> jax.Array:
    tc, _, _, z = previous_position.T

    modulation = cutoff((tc - z) / c, pulse_parameters.phi_0, pulse_parameters.tau_0)
    modulation = jnp.expand_dims(modulation, axis=-1)

    electric_field, magnetic_field = plane_wave_fields(
        laser_parameters, previous_position
    )

    electric_field = modulation * electric_field
    magnetic_field = modulation * magnetic_field

    acceleration = compute_acceleration(
        previous_momentum,
        electric_field,
        magnetic_field,
        charge_to_mass_ratio=-1,
    )

    # if (check_for_errors)
    # {
    #     check_integration_results(previous_momentum, acceleration);
    # }

    return previous_momentum + time_step * acceleration


def integrate_particle(
    initial_position: jax.Array,
    initial_momentum: jax.Array,
    start_time: jdc.Static[float],
    end_time: jdc.Static[float],
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaserParameters],
    pulse_parameters: jdc.Static[PulseParameters],
) -> tuple[jax.Array, jax.Array]:
    previous_position = initial_position
    previous_momentum = initial_momentum

    time_steps = np.arange(start_time, end_time, time_step)

    for _proper_time in time_steps:
        new_momentum = compute_new_momentum(
            previous_position,
            previous_momentum,
            time_step,
            laser_parameters,
            pulse_parameters,
        )
        new_position = previous_position + time_step * new_momentum

        # TODO: sum up scattered radiation

        previous_position = new_position
        previous_momentum = new_momentum

    return new_position, new_momentum


def simulate_trajectories(
    initial_positions: jax.Array,
    initial_momenta: jax.Array,
    start_time: jdc.Static[float],
    end_time: jdc.Static[float],
    time_step: jdc.Static[float],
    laser_parameters: jdc.Static[LaserParameters],
    pulse_parameters: jdc.Static[PulseParameters],
) -> IntegrationResult:
    num_devices = len(jax.devices())

    mesh = jax.make_mesh((num_devices,), ("i",))
    with jax.set_mesh(mesh):
        initial_positions = jax.reshard(initial_positions, P("i", None))
        initial_momenta = jax.reshard(initial_momenta, P("i", None))

        final_positions, final_momenta = integrate_particle(
            initial_positions,
            initial_momenta,
            start_time,
            end_time,
            time_step,
            laser_parameters,
            pulse_parameters,
        )

    return IntegrationResult(final_positions, final_momenta)


@jdc.jit
def compute_angular_momentum(
    positions: jax.Array, momenta: jax.Array, particle_mass: jdc.Static[float]
) -> jax.Array:
    return particle_mass * (
        positions[:, 1] * momenta[:, 2] - positions[:, 2] * momenta[:, 1]
    )


def main() -> None:
    print("Enabling float64 support in JAX")
    jax.config.update("jax_enable_x64", True)

    print("Enabling CPU parallelism in JAX")
    num_cpus = multiprocessing.cpu_count()
    jax.config.update("jax_num_cpu_devices", min(128, num_cpus))

    # print("Available JAX devices:", jax.local_devices())

    jax.config.update("jax_compilation_cache_dir", "jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    # Monochromatic laser
    # ~800 nm, red light
    laser_frequency = 0.057

    a_0 = 1
    m_e = 1
    q = -1

    amplitude = a_0 * m_e * c * laser_frequency / abs(q)
    polarization = Polarization(1.0, 0.0)

    laser_parameters = LaserParameters(laser_frequency, amplitude, polarization)

    seed = 42
    generator = np.random.default_rng(seed)
    num_electrons = 8192

    print(f"Working with {num_electrons} electrons")

    radial_index = 0
    waist_radius = 75 * laser_parameters.wavelength

    disk_radius = (1.75 + radial_index) * waist_radius

    initial_positions = generate_initial_positions_on_disk(
        generator, disk_radius, num_electrons
    )
    # Add a 0 on the first index to obtain position 4-vectors
    initial_positions = np.concatenate(
        (np.zeros((num_electrons, 1), dtype=np.float64), initial_positions), axis=-1
    )

    # TODO: give the electrons initial velocities and check what is
    # the Doppler-shifted frequency of the scattered radiation (should be ~ 4 \gamma^2)

    initial_momenta = np.zeros((num_electrons, 4), dtype=np.float64)
    # u0 = c \gamma
    # We set \gamma to be 1 initially
    initial_momenta[:, 0] = c

    initial_positions = jnp.asarray(initial_positions)
    initial_momenta = jnp.asarray(initial_momenta)

    ### Pulse parameters ###
    tau_0 = 10 / laser_frequency
    phi_0 = 3 * tau_0

    pulse_parameters = PulseParameters(phi_0, tau_0)

    integration_start_time = 0.0
    integration_end_time = 6 * tau_0
    integration_duration = integration_end_time - integration_start_time

    time_step = integration_duration / 1000

    print(
        f"Integrating from t = {integration_start_time} to t = {integration_end_time}, with a time step of dt = {time_step}"
    )

    start_time = perf_counter()

    result = simulate_trajectories(
        initial_positions,
        initial_momenta,
        integration_start_time,
        integration_end_time,
        time_step,
        laser_parameters,
        pulse_parameters,
    )

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Took {duration} seconds")

    print("Plotting results")

    plots_directory = Path("plots")
    plots_directory.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    colors = result.final_momenta[:, 3]

    points = ax.scatter(
        initial_positions[:, 1] / waist_radius,
        initial_positions[:, 2] / waist_radius,
        s=3,
        c=colors,
    )

    fig.colorbar(points)

    fig.tight_layout()
    fig.savefig(plots_directory / "final_momenta.png")

    fig = plt.figure()

    angular_momenta = compute_angular_momentum(
        result.final_positions, result.final_momenta, m_e
    )

    plot_angular_momentum_distribution(
        fig,
        np.asarray(initial_positions[:, 1:4]),
        waist_radius,
        np.asarray(angular_momenta),
    )

    fig.tight_layout()
    fig.savefig(plots_directory / "final_angular_momenta.png")


if __name__ == "__main__":
    typer.run(main)
