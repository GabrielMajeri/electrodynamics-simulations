#pragma once

#include <cassert>
#include <utility>
#include <vector>

#include "beam.h++"
#include "vector.h++"

std::pair<std::vector<Position>, std::vector<Momentum>> integrate_trajectories(
    LaguerreGaussBeamParameters parameters,
    Real charge_to_mass_ratio,
    Real phi_0, Real tau_0,
    std::vector<Position> initial_positions,
    std::vector<Momentum> initial_momenta,
    Real integration_start_time, Real integration_end_time,
    Real time_step);

OPENACC_ROUTINE
Acceleration compute_acceleration(
    Momentum previous_momentum, Real charge_to_mass_ratio,
    Vector3D electric_field, Vector3D magnetic_field);
