#pragma once

#include <vector>

#include "types.h++"
#include "vector.h++"

std::vector<Real> compute_angular_momenta_in_z_direction(
    Real particle_mass,
    std::vector<Position> positions, std::vector<Momentum> momenta);
