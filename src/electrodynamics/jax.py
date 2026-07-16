import os

import jax


def initialize_jax() -> None:
    "Initializes and configures JAX."

    # BUG: disable NVLink-related stuff; it gives error on our system
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    # Disable memory pre-allocation, since it's a shared system
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    print("Enabling float64 support in JAX")
    jax.config.update("jax_enable_x64", True)

    # print("Enabling CPU parallelism in JAX")
    # num_cpus = multiprocessing.cpu_count()
    # jax.config.update("jax_num_cpu_devices", min(128, num_cpus))

    # print("Available JAX devices:", jax.local_devices())

    jax.config.update("jax_compilation_cache_dir", "jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
