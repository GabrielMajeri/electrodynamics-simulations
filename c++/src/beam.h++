#pragma once

#include "common.h++"
#include "polarization.h++"
#include "types.h++"
#include "vector.h++"

/// @brief Parameters controlling the shape and time evolution of a Laguerre-Gauss beam.
struct LaguerreGaussBeamParameters
{
    // E_0 = B_0
    Real amplitude;
    // \xi
    PolarizationVector polarization;
    // w(0)
    Real waist_radius;
    // \lambda
    Real wavelength;
    // \omega
    Real angular_velocity;
    // p
    uint32_t radial_index;
    // l (or m)
    int32_t azimuthal_index;
};

OPENACC_ROUTINE
std::pair<Vector3D, Vector3D> laguerre_gauss_beam_electric_and_magnetic_field(
    LaguerreGaussBeamParameters parameters, Vector3D position, Real time);

OPENACC_ROUTINE
Real laguerre_polynomial(uint32_t n, Real alpha, Real x);

OPENACC_ROUTINE
Real cutoff(Real phi, Real phi_0, Real tau_0);
