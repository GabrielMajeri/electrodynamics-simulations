#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "vector.h++"

std::vector<Position> generate_initial_electron_positions_on_circle(std::size_t num_electrons, double circle_radius);

std::vector<Position> generate_initial_electron_positions_within_disk(std::size_t num_electrons, double disk_radius, uint32_t seed);

std::vector<Momentum> generate_initial_electron_momenta(std::size_t num_electrons);
