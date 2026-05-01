#include <chrono>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

using Real = double;

// The mathematical constant $\pi$.
constexpr Real pi = std::numbers::pi;

struct Position
{
    Real x, y, z;
};

std::vector<Position> generate_initial_electron_positions(size_t num_electrons, double disk_radius, uint32_t seed);

int main()
{
    std::cout << "Starting Laguerre-Gauss beam angular momentum transfer simulation code" << std::endl;

    // constexpr size_t num_electrons = 16 * 1024;
    constexpr size_t num_electrons = 64 * 1024;
    constexpr Real c = 137.036;
    constexpr Real omega = 0.057;
    constexpr Real lambda = 2 * pi * c / omega;
    constexpr Real waist_radius = 75 * lambda;

    constexpr uint32_t radial_index = 2;

    constexpr Real disk_radius = (1.75 + radial_index) * waist_radius;

    constexpr uint32_t seed = 42;

    std::cout << "Generating initial positions for " << num_electrons << " electrons, uniformly distributed within a disk of radius " << disk_radius << " in the x-y plane, centered at the origin" << std::endl;

    const auto start = std::chrono::steady_clock::now();

    const auto initial_electron_positions = generate_initial_electron_positions(num_electrons, disk_radius, seed);

    const auto finish = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds = finish - start;

    std::cout << "Generated " << num_electrons << " initial positions in " << elapsed_seconds << " seconds" << std::endl;

    // TODO: simulate electron's motion in a Laguerre-Gauss beam field

    return 0;
}

std::vector<Position> generate_initial_electron_positions(std::size_t num_electrons, double disk_radius, uint32_t seed)
{
    std::vector<Position> positions(num_electrons);

    const double disk_radius_squared = disk_radius * disk_radius;

    std::uniform_real_distribution<Real>
        unif_r(0.0, disk_radius_squared),
        unif_angle(0.0, 2 * pi);

    std::default_random_engine rng(seed);

#pragma omp parallel for
    for (size_t i = 0; i < num_electrons; ++i)
    {
        const double r = std::sqrt(unif_r(rng));
        const double theta = unif_angle(rng);

        const Real
            x = r * std::cos(theta),
            y = r * std::sin(theta),
            z = 0;

        positions[i] = Position{x, y, z};
    }

    return positions;
}
