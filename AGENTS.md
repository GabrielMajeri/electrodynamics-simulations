# AGENTS.md

Operational guide for AI coding agents working in `electrodynamics-simulations` (Python 3.14+ / JAX relativistic electrodynamics and laser-electron OAM scattering).

## Commands

- **Install / sync environment** (pick one backend extra: `cpu`, `cu130`, `rocm`):
  `uv sync --extra cpu`
- **Run test suite**:
  `.venv/bin/pytest -v` (or `uv run pytest -v` when `uv` is on `PATH`)
- **Run a simulation or diagnostic script**:
  `.venv/bin/python scripts/<script>.py --help`
- **Type checking**: `basedpyright` in `basic` mode (see `.vscode/settings.json`).

## Architecture & Boundaries

- Place reusable physics, beam/pulse models, integrators, detectors, OAM/energy diagnostics, plotting, and serialization in `src/electrodynamics/`. Keep `scripts/` as thin Typer CLI experiment drivers and offline analysis tools.
- **Do not edit** `old_codes/` unless explicitly instructed; they are archived implementations.
- **Do not commit** simulation outputs in `runs/` or XLA caches in `jax_cache/`.

## Physics & Numerical Guardrails

Read [`docs/conventions.md`](docs/conventions.md) before touching any field, trajectory, detector, OAM, or energy calculation. Key non-obvious constraints:

- **Float64 is mandatory**: Always call `initialize_jax()` in scripts and use the `jax_config` fixture in JAX tests (`jax_enable_x64 = True`). Float32 breaks retarded-time phase cancellation in `fields.py`.
- **Two distinct $\mathbf{B}$-field normalizations**:
  - `src/electrodynamics/beams.py` returns $\mathbf{B}_{\mathrm{beam}} = \mathbf{B}_{\mathrm{Gauss}} / c$ (matching the Lorentz force without $1/c$).
  - `src/electrodynamics/fields.py` returns Gaussian-unit scattered Fourier fields with $|\tilde{\mathbf{B}}_{\mathrm{Gauss}}| = |\tilde{\mathbf{E}}|$. Never mix the two without explicit $c$ scaling.
- **Intentional beam normalization difference**: `LaguerreGaussBeamParameters` includes $\sqrt{2\,p!/(p+|\ell|)!}$ ($\iint |u|^2 dA = \pi w_0^2 E_0^2$, matching Hu et al. 2018), whereas `GaussianBeamParameters` omits $\sqrt{2}$ ($\iint |u|^2 dA = \frac{\pi}{2} w_0^2 E_0^2$). Do not change existing beam amplitudes; account for both in energy formulas.
- **OAM integration & sign**: Never use raw `np.sum` on `dL_z`; always integrate over detector area with `integrate_detector_oam_spectral_density`. Distinguish lab-frame $L_z$ (`propagation_sign=1`) from reflected beam-axis helicity (`propagation_sign=-1`) in `compute_oam_per_photon`.
- **Proper-time vs. lab-time Liénard-Wiechert flag**: Match `integration_step_rk4` with `is_lab_time=False` and `integration_step_rk4_lab_time` with `is_lab_time=True`. Pass `initial_momentum` when particles start with nonzero drift to suppress boundary-window radiation.
- **JAX static typing & sharding**: Annotate compile-time config dataclasses/flags with `jdc.Static[...]`. Scripts using `iterate_sharded_particle_batches` require `num_particles % batch_size == 0` and `batch_size % len(jax.devices()) == 0`.

## Code Style & Verification

- Target Python 3.14+ (`X | Y` unions, PEP 695 generics, `pathlib.Path`, Typer `Annotated` CLI options).
- After edits, run `.venv/bin/pytest -v` and verify physics changes against analytical invariants (4-momentum mass shell $u_0^2 - |\mathbf{u}|^2 = (m_e c)^2$, zero scattered field for constant velocity, exact OAM per photon, Parseval energy flux).

## Documentation

- [`docs/conventions.md`](docs/conventions.md) — Units, field normalizations, OAM sign conventions, and analytical benchmarks.
- [`docs/architecture.md`](docs/architecture.md) — Module layout, CLI scripts, and JAX sharding model.
