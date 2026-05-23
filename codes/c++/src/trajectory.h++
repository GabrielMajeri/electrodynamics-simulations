#pragma once

#include <vector>

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
