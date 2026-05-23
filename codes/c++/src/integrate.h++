#pragma once

#include "vector.h++"

OPENACC_ROUTINE
std::pair<Position, Momentum> perform_integration_step(
    Position previous_position, Momentum previous_momentum);

OPENACC_ROUTINE
Acceleration compute_acceleration(
    Momentum previous_momentum, Vector3D electric_field, Vector3D magnetic_field);
