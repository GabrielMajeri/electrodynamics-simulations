import os
from collections.abc import Iterator
from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def initialize_jax(device: Literal["cpu", "cuda"] = "cuda") -> None:
    "Initializes and configures JAX."

    print("Enabling float64 support in JAX")
    jax.config.update("jax_enable_x64", True)

    print("Enabling JAX compilation caching")
    jax.config.update("jax_compilation_cache_dir", "jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    if device == "cpu":
        print("Configuring JAX to use the CPU")

        import multiprocessing

        num_cpus = multiprocessing.cpu_count()
        jax.config.update("jax_num_cpu_devices", min(128, num_cpus))

    elif device == "cuda":
        print("Configuring JAX to run on CUDA")

        # BUG: disable NVLink-related stuff; it gives error on our system
        os.environ["NCCL_NVLS_ENABLE"] = "0"

        # Disable memory pre-allocation, since it's a shared system
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )

    print("Available JAX devices:", jax.local_devices())


def iterate_sharded_particle_batches(
    initial_positions: jax.Array,
    initial_momenta: jax.Array,
    batch_size: int,
) -> Iterator[tuple[int, int, jax.Array, jax.Array]]:
    """Validates batch/device divisibility, activates a 1D JAX device mesh,
    and yields `(batch_index, num_batches, sharded_positions, sharded_momenta)`.
    """
    num_particles = len(initial_positions)
    num_devices = min(num_particles, len(jax.devices()))
    if num_particles % batch_size != 0:
        raise ValueError(
            f"Number of electrons ({num_particles}) must be divisible by batch_size ({batch_size})"
        )
    if batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by number of devices ({num_devices})"
        )

    mesh = jax.make_mesh((num_devices,), ("i",))
    num_batches = num_particles // batch_size
    position_batches = jnp.split(initial_positions, num_batches, axis=0)
    momentum_batches = jnp.split(initial_momenta, num_batches, axis=0)

    with jax.set_mesh(mesh):
        for index, (pos_batch, mom_batch) in enumerate(
            zip(position_batches, momentum_batches, strict=True)
        ):
            yield (
                index,
                num_batches,
                jax.device_put(pos_batch, P("i", None)),
                jax.device_put(mom_batch, P("i", None)),
            )
