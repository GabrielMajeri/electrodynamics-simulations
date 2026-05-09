#pragma once

#include <cstdint>

#include "common.h++"
#include "polarization.h++"
#include "types.h++"
#include "vector.h++"

OPENACC_ROUTINE
std::pair<Vector3D, Vector3D> laguerre_gauss_beam_electric_and_magnetic_field(
    Vector3D position, Real time);

OPENACC_ROUTINE
Real laguerre_polynomial(std::uint32_t n, Real alpha, Real x);

OPENACC_ROUTINE
Real cutoff(Real phi, Real phi_0, Real tau_0);
