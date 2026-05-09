#pragma once

#include <cassert>
#include <utility>
#include <vector>

#include "beam.h++"
#include "vector.h++"

std::pair<std::vector<Position>, std::vector<Momentum>> integrate_trajectories(
    std::vector<Position> initial_positions, std::vector<Momentum> initial_momenta);

OPENACC_ROUTINE
Acceleration compute_acceleration(
    Momentum previous_momentum, Vector3D electric_field, Vector3D magnetic_field);
