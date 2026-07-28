#include "initial_conditions.h++"

#include "constants.h++"

#include <cmath>

std::vector<Position> generate_initial_electron_positions_on_circle(std::size_t num_electrons, double circle_radius)
{
    std::vector<Position> positions(num_electrons);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        const Real theta = 2 * pi * static_cast<Real>(i) / num_electrons;
        const Real
            x = circle_radius * std::cos(theta),
            y = circle_radius * std::sin(theta),
            z = 0;

        positions[i] = Position{0, x, y, z};
    }

    return positions;
}

std::vector<Position> generate_initial_electron_positions_within_disk(size_t num_electrons, double disk_radius, uint32_t seed)
{
    std::vector<Position> positions(num_electrons);

    // Use the inverse sampling method to generate points uniformly in the disk
    const double disk_radius_squared = disk_radius * disk_radius;

    std::uniform_real_distribution<Real>
        unif_r(0.0, disk_radius_squared),
        unif_angle(0.0, 2 * pi);

    std::default_random_engine rng(seed);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        // Generate a new position for each electron
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

std::vector<Momentum> generate_initial_electron_momenta_stationary(size_t num_electrons)
{
    std::vector<Momentum> momenta(num_electrons);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        momenta[i].u0 = c;
    }

    return momenta;
}

std::vector<Momentum> generate_initial_electron_momenta_random_velocity(
    std::size_t num_electrons, uint32_t seed)
{
    std::vector<Momentum> momenta(num_electrons);

    // Maximum relativistic factor of generated electrons
    constexpr Real max_gamma = 1000;
    static_assert(max_gamma >= 1, "Lorentz factor must be greater than or equal to unity");

    constexpr Real max_gamma_squared = max_gamma * max_gamma;

    const Real max_beta = std::sqrt((max_gamma_squared - 1) / max_gamma_squared);

    const Real max_velocity = max_beta * c;

    std::uniform_real_distribution<Real> unif_u(0.0, 1);

    std::default_random_engine rng(seed);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        // Random velocity norm
        const auto v = max_velocity * unif_u(rng);

        const auto u_1 = unif_u(rng);
        const auto u_2 = unif_u(rng);

        const auto phi = std::acos(2 * u_1 - 1) - pi / 2;
        const auto theta = 2 * pi * u_2;

        const auto cos_phi = std::cos(phi);

        // Random point on sphere
        const Vector3D direction{cos_phi * std::cos(theta), cos_phi * std::sin(theta), std::sin(phi)};

        const auto beta = v / c;

        momenta[i].u1 = v * direction.x;
        momenta[i].u2 = v * direction.y;
        momenta[i].u3 = v * direction.z;
        momenta[i].u0 = c / sqrt(1 - beta * beta);
    }

    return momenta;
}
