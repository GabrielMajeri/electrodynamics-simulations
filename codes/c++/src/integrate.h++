#pragma once

#include <vector>

#include "vector.h++"

OPENACC_ROUTINE
std::pair<Position, Momentum> perform_integration_step_euler(
    Position previous_position, Momentum previous_momentum);

OPENACC_ROUTINE
std::pair<Position, Momentum> perform_integration_step_rk4(
    Position previous_position, Momentum previous_momentum);

OPENACC_ROUTINE
Acceleration compute_acceleration(
    Momentum previous_momentum, Vector3D electric_field, Vector3D magnetic_field);

OPENACC_ROUTINE
void integrate_scattered_field(
    Real current_time,
    const Position &position, const Momentum &momentum,
    const Position &initial_position,
    const std::vector<Vector3D> &detector_positions,
    std::vector<ComplexVector3D> &electric_field,
    std::vector<ComplexVector3D> &magnetic_field);
