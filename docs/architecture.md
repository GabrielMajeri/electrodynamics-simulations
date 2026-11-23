# Codebase Architecture

This document describes the current structure of `electrodynamics-simulations`, its library modules, simulation scripts, and JAX execution model. For units and field conventions, see [`docs/conventions.md`](conventions.md).

---

## 1. Repository Layout

```text
electrodynamics-simulations/
├── src/electrodynamics/   # Core JAX/NumPy physics, integrator, detector, and plotting library
├── scripts/               # Typer CLI simulation drivers and post-processing diagnostics
├── tests/                 # Pytest unit and physics verification suite
├── docs/                  # Reference documentation (conventions, architecture)
├── runs/                  # Gitignored simulation outputs (configs, arrays, PDF/PNG plots)
├── jax_cache/             # Gitignored persistent XLA/JAX JIT compilation cache
├── old_codes/             # Archived prototypes in C++, C#, Fortran, Haskell, Julia, Python, Rust
```

---

## 2. Core Library (`src/electrodynamics/`)

| Module | Responsibility | Key Public Symbols |
| --- | --- | --- |
| [`constants.py`](../src/electrodynamics/constants.py) | Atomic unit constants | `SPEED_OF_LIGHT`, `ELECTRON_MASS`, `ELECTRON_CHARGE`, `HARTREE_TO_EV`, `ELECTRON_REST_ENERGY_MEV`, `DEFAULT_LASER_FREQUENCY` |
| [`polarization.py`](../src/electrodynamics/polarization.py) | Unit-norm Jones vectors & spin-OAM helpers | `Polarization`, `Polarizations`, `get_polarization_by_name`, `resolve_circular_polarization` |
| [`beams.py`](../src/electrodynamics/beams.py) | Analytic plane-wave, Gaussian, and Laguerre-Gauss EM fields (`B_beam = B_Gauss/c`) | `LaserParameters`, `GaussianBeamParameters`, `LaguerreGaussBeamParameters`, `compute_beam_fields`, `compute_directional_beam_fields`, `compute_oblique_beam_fields` |
| [`pulse.py`](../src/electrodynamics/pulse.py) | Temporal envelope modulation ($g(\phi)$) | `GaussianPulseParameters`, `CosineSquaredPulseParameters`, `PulseWithFlatPeakParameters`, `compute_pulse_envelope` |
| [`fields.py`](../src/electrodynamics/fields.py) | Lorentz derivatives, relativistic kinematics, Liénard-Wiechert Fourier fields | `compute_relativistic_reflection_parameters`, `compute_doppler_shift_factor`, `compute_acceleration_of_charged_particle_in_em_field`, `compute_lab_time_lorentz_derivatives`, `compute_scattered_electric_and_magnetic_fields`, `compute_coherent_detector_and_spectrum_fields` |
| [`integrate.py`](../src/electrodynamics/integrate.py) | Proper-time and lab-time RK4 steppers (single and two-pulse) + snapshot schedules | `SnapshotData`, `parse_snapshot_schedule`, `integration_step_rk4`, `integration_step_rk4_lab_time`, `integration_step_rk4_lab_time_two_pulses` |
| [`initial_conditions.py`](../src/electrodynamics/initial_conditions.py) | Particle bunch spatial & relativistic momentum generators | `to_position_four_vectors`, `generate_initial_positions_uniformly_on_disk`, `generate_initial_positions_uniformly_within_ball`, `generate_initial_positions_on_thin_sheet`, `generate_nanofiber_initial_positions`, `generate_initial_particle_momenta_moving_towards_laser`, `generate_initial_particle_momenta_at_rest` |
| [`detector.py`](../src/electrodynamics/detector.py) | Rectangular detector grids and local frame projections | `DetectorParameters`, `initialize_detector_positions_negative_z`, `initialize_detector_positions`, `initialize_detector_positions_reflected`, `project_fields_to_detector_frame` |
| [`angular_momentum.py`](../src/electrodynamics/angular_momentum.py) | Particle $L_z$, EM OAM spectral density, area integration, and OAM per photon | `compute_particle_angular_momentum_z`, `compute_angular_momentum_derivative`, `integrate_detector_oam_spectral_density`, `compute_oam_per_photon` |
| [`jax.py`](../src/electrodynamics/jax.py) | Float64 setup, persistent cache, CPU/CUDA config, and particle batch sharding | `initialize_jax`, `iterate_sharded_particle_batches` |
| [`plotting.py`](../src/electrodynamics/plotting.py) | Matplotlib setup, trajectory, phase-space, spectrum, and detector OAM figures | `save_figure`, `plot_experimental_setup`, `plot_detector_radiation_and_oam`, `save_detector_field_component_plots`, `save_nanofiber_snapshot_plots`, `save_nanofiber_final_plots` |
| [`serialization.py`](../src/electrodynamics/serialization.py) | Run directory layout and JSON config encoding | `ConfigurationEncoder`, `prepare_output_directory`, `save_json` |

---

## 3. Simulation & Diagnostic Scripts (`scripts/`)

All scripts are Typer CLIs writing outputs to `runs/<experiment_name>/...`.

| Script | Physics / Purpose | Integrator | Detector / Key Outputs |
| --- | --- | --- | --- |
| [`flying_mirror_reflection.py`](../scripts/flying_mirror_reflection.py) | LG beam reflection off a thin relativistic electron sheet (normal or oblique, Chen et al. 2022) | Lab-time RK4 (`shard_map`) | Reflected 2D rectangular detector + spectrum; saves `.npy` fields, `dL_z`, summary JSON, plots |
| [`oam_transfer_study.py`](../scripts/oam_transfer_study.py) | LG beam scattering off disk, thin-sheet, ball, or nanofiber electron bunches | Proper-time RK4 (`shard_map`) | Fixed rectangular detector at `-25000 lambda_0` + spectrum; saves config and `.pdf` plots |
| [`doppler_shift_study.py`](../scripts/doppler_shift_study.py) | Frequency spectrum around the relativistic Doppler-upshifted peak | Proper-time RK4 (`shard_map`) | Single spectrum observer point; saves `electric_field_spectrum.pdf` |
| [`single_electron_trajectory.py`](../scripts/single_electron_trajectory.py) | Single-electron trajectory and Liénard-Wiechert integrand diagnostics | Proper-time RK4 | Records full trajectory, oscillatory kernel, and field multipliers as `.pdf` plots |
| [`nanofiber_attosecond_bunches.py`](../scripts/nanofiber_attosecond_bunches.py) | Circular LG pulse generating helical/disk microbunches from a nanofiber (Hu et al. 2018) | Lab-time RK4 + snapshots | Forward rectangular detector (`+z`); saves `.npy` detector/spectrum data, snapshots, OAM history |
| [`two_stage_nanofiber_probe_collision.py`](../scripts/two_stage_nanofiber_probe_collision.py) | Drive LG nanofiber bunching (`+z`) + counter-propagating probe collision (`-z`) | Lab-time RK4 (2-pulse) | Forward high-frequency detector + snapshots; saves `.npy` fields, `dL_z`, summary JSON |
| [`extended_oam_diagnostics.py`](../scripts/extended_oam_diagnostics.py) | Post-processor for `flying_mirror` runs quantifying OAM/LM non-collinearity | Offline analysis | Reads saved `.npy` / `configuration.json`; generates tilt geometry and transverse OAM plots |

---

## 4. JAX Execution & Sharding Pattern

1. **Initialization**: [`initialize_jax()`](../src/electrodynamics/jax.py#L10) enables 64-bit floating-point precision (`jax_enable_x64 = True`), configures the persistent JIT cache under `jax_cache/`, and sets CPU device count or CUDA environment flags (`NCCL_NVLS_ENABLE=0`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`).
2. **Static vs. Dynamic Arguments**: Beam, pulse, and detector parameter dataclasses use `@jdc.pytree_dataclass` and are marked `jdc.Static[...]` on `@jdc.jit` / `@jax.shard_map` boundaries so JAX specializes kernels without tracing static Python branches (`exact_paraxial`, `radial_index`, etc.).
3. **Batch & Multi-Device Sharding**: [`iterate_sharded_particle_batches()`](../src/electrodynamics/jax.py#L46) splits particles into batches and distributes each batch across a 1D device mesh `("i",)` via `PartitionSpec("i", None)` and `jax.shard_map`. Callers must satisfy:
   - `num_particles % batch_size == 0`
   - `batch_size % num_devices == 0`
4. **Accumulator Pattern**: Inside `jax.lax.scan`, each RK4 step updates particle `(positions, momenta)` and coherently accumulates per-particle Liénard-Wiechert contributions (`dt * E`, `dt * B`), which are summed across particles and batches on the host after each batch completes.

---

## 5. Run Serialization

[`prepare_output_directory()`](../src/electrodynamics/serialization.py#L48) creates outputs under:

- `runs/<experiment_name>/<run_tag>/` when `run_tag` is non-empty,
- `runs/<experiment_name>/<overwrite_run_tag>/` when `overwrite=True` (default `latest`),
- `runs/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/` (with optional `timestamp_suffix`) otherwise.

[`ConfigurationEncoder`](../src/electrodynamics/serialization.py#L15) serializes `jax_dataclasses`, `complex`, `numpy`, and `jax.Array` objects to JSON while automatically annotating `waist_radius_in_multiples_of_lambda` and `phi_0_in_multiples_of_tau_0`.
