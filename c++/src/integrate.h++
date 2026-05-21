#pragma once

#include <cassert>
#include <utility>
#include <vector>

#include "beam.h++"
#include "vector.h++"

struct IntegrationResult
{
    std::vector<Position> positions;
    std::vector<Momentum> momenta;

    std::vector<Position> particle_trajectory;

    std::vector<Vector3D> detector_positions;
    std::vector<ComplexVector3D> electric_field;
    std::vector<ComplexVector3D> magnetic_field;
};

IntegrationResult analytic_trajectories(std::vector<Position> initial_positions);

IntegrationResult integrate_trajectories(
    std::vector<Position> initial_positions, std::vector<Momentum> initial_momenta);

OPENACC_ROUTINE
std::pair<Position, Momentum> perform_integration_step(
    Position previous_position, Momentum previous_momentum);

OPENACC_ROUTINE
Acceleration compute_acceleration(
    Momentum previous_momentum, Vector3D electric_field, Vector3D magnetic_field);
