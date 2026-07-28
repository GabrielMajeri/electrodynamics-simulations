# Numerical simulation of emitted radiation from accelerated charges in a laser field

## Description

This repository contains numerical codes for computing the angular momentum transferred to a bunch of charged particles (electrons) and the radiation scattered by them, when they are under the influence of a Laguerre-Gauss beam (from a laser, for example).

They are implemented in [Python](https://www.python.org/), using [JAX](https://docs.jax.dev/en/latest/) for high-performance numerical linear algebra (including GPU support).

## Features

- Analytic formulas for plane waves, Gaussian and Laguerre-Gauss beams
- Initial conditions generation (uniformly distributed within disk, ball etc)
- Relativistic trajectory integration (by computing the electromagnetic field tensor)
- Computation of angular momentum transfer
- Computation of scattered electric and magnetic fields

## Contents

Most of the code is available as a library, which can be found in `src/electrodynamics`. See the `scripts` directory for some example use cases.

We also have some implementations written in various programming languages, which can be found in the [`old_codes`](old_codes) directory. Since Python with JAX is so fast, we stopped working on the alternatives.

## Credits

Gabriel Majeri and Mădălina Boca,
University of Bucharest
