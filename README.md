# Numerical simulation of emitted radiation from accelerated charges in a laser field

## Description

This repository contains numerical codes for computing the angular momentum transferred to a bunch of charged particles (electrons) and the radiation scattered by them, when they are under the influence of a Laguerre-Gauss beam (from a laser, for example).

They are implemented in [Python](https://www.python.org/), using [JAX](https://docs.jax.dev/en/latest/) for high-performance numerical linear algebra (including GPU support; tested with both NVIDIA CUDA and AMD ROCm).

## Features

- Analytic formulas for plane waves, Gaussian and Laguerre-Gauss beams
- Initial conditions generation (uniformly distributed within disk, ball etc)
- Relativistic trajectory integration (by computing the electromagnetic field tensor)
- Computation of angular momentum transfer
- Computation of scattered electric and magnetic fields

## Contents

- [`src/electrodynamics`](src/electrodynamics): Core simulation library (beams, pulses, trajectories, scattered fields, detectors, OAM, plotting, and serialization).
- [`scripts`](scripts): CLI simulation drivers and post-processing diagnostic tools.
- [`tests`](tests): Automated pytest unit and physics verification suite.
- [`docs`](docs): Reference documentation on units, field conventions, and codebase architecture.
- [`old_codes`](old_codes): Earlier prototypes in C++, C#, Fortran, Haskell, Julia, Python, and Rust (kept for reference; superseded by Python + JAX).

## Getting Started

Install dependencies using [`uv`](https://docs.astral.sh/uv/) with the desired JAX backend extra (`cpu`, `cu130`, or `rocm`):

```bash
uv sync --extra cpu
```

Run the test suite and inspect script CLI options:

```bash
uv run pytest -v
uv run python scripts/flying_mirror_reflection.py --help
```

## Documentation

- [Physics & Unit Conventions](docs/conventions.md) — Atomic units, $\mathbf{B}$-field and beam normalizations, Parseval spectral energy flux, OAM sign conventions, and analytical benchmarks.
- [Codebase Architecture](docs/architecture.md) — Library modules, CLI scripts, JAX float64/sharding execution model, and run serialization.
- [Agent Guidelines](AGENTS.md) — Concise repository guide for AI coding assistants (following the [AGENTS.md](https://agents.md/) specification).

## Credits

Gabriel Majeri and Mădălina Boca,
University of Bucharest

## AI usage

While the initial ideas and code was written by humans, we've since come to rely heavily on Gemini and Claude for developing and refining the code.
