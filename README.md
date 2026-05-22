# Numerical simulation of emitted radiation from accelerated charges in a laser field

## Description

This repository contains numerical codes for computing the angular momentum transferred to a bunch of charged particles (electrons) and the radiation scattered by them, when they are under the influence of a Laguerre-Gauss beam (from a laser, for example).

## Features

- Analytic formulas for plane waves, Gaussian and Laguerre-Gauss beams
- Initial conditions generation
- Relativistic trajectory integration (by computing the electromagnetic field tensor)
- Computation of angular momentum transfer
- Computation of scattered electric and magnetic fields

## Contents

The programs (written in various programming languages) can be found in the [`codes`](codes) directory. The Python implementation is very pedagogical, but fairly slow. The C++, Fortran and Rust implementations are usually the most feature-complete. The Haskell and Julia implementations are experimental, while the C# version is incomplete and unlikely to be continued.

## Credits

Gabriel Majeri and Mădălina Boca,
University of Bucharest
