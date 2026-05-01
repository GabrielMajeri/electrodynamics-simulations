#include <cassert>
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
    Real t, x, y, z;
};

struct Momentum
{
    Real gamma, vx, vy, vz;
};

std::vector<Position> generate_initial_electron_positions(std::size_t num_electrons, double disk_radius, uint32_t seed);
std::vector<Momentum> generate_initial_electron_momenta(std::size_t num_electrons);
std::pair<std::vector<Position>, std::vector<Momentum>> integrate_trajectories(
    std::vector<Position> initial_positions,
    std::vector<Momentum> initial_momenta,
    Real integration_start_time, Real integration_end_time,
    Real time_step);

int main()
{
    std::cout << "Starting Laguerre-Gauss beam angular momentum transfer simulation code" << std::endl;

    constexpr size_t num_electrons = 16 * 1024;
    // constexpr size_t num_electrons = 64 * 1024;
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

    const auto initial_electron_momenta = generate_initial_electron_momenta(num_electrons);

    constexpr auto tau_0 = 10 / omega;

    constexpr Real start_time = 0.0, end_time = 6 * tau_0;
    constexpr Real time_step = 0.1;
    const auto [final_positions, final_momenta] = integrate_trajectories(initial_electron_positions, initial_electron_momenta, start_time, end_time, time_step);

    // TODO: simulate electron's motion in a Laguerre-Gauss beam field

    return 0;
}

std::vector<Position> generate_initial_electron_positions(size_t num_electrons, double disk_radius, uint32_t seed)
{
    std::vector<Position> positions(num_electrons);

    const double disk_radius_squared = disk_radius * disk_radius;

    std::uniform_real_distribution<Real>
        unif_r(0.0, disk_radius_squared),
        unif_angle(0.0, 2 * pi);

    std::default_random_engine rng(seed);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        const double r = std::sqrt(unif_r(rng));
        const double theta = unif_angle(rng);

        const Real
            x = r * std::cos(theta),
            y = r * std::sin(theta),
            z = 0;

        positions[i] = Position{0, x, y, z};
    }

    return positions;
}

std::vector<Momentum> generate_initial_electron_momenta(size_t num_electrons)
{
    std::vector<Momentum> momenta(num_electrons);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        momenta[i].gamma = 1;
    }

    return momenta;
}

std::pair<std::vector<Position>, std::vector<Momentum>> integrate_trajectories(
    std::vector<Position> initial_positions, std::vector<Momentum> initial_momenta,
    Real integration_start_time, Real integration_end_time,
    Real time_step)
{
    size_t num_particles = initial_positions.size();
    assert(num_particles == initial_momenta.size());

    std::vector<Position> current_positions(num_particles);
    std::vector<Momentum> current_momenta(num_particles);

    Real integration_duration = integration_end_time - integration_start_time;
    size_t num_steps = integration_duration / time_step;

    Real current_time = 0;
    for (size_t step = 0; step <= num_steps; ++step)
    {
        // TODO: evaluate Laguerre-Gauss beam formula at previous position
        // TODO: compute electromagnetic field tensor and use it to determine the acceleration

        current_time += time_step;
    }

    return std::make_pair(current_positions, current_momenta);
}
