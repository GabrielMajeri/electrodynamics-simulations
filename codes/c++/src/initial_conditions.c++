#include "initial_conditions.h++"

#include "constants.h++"

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
        momenta[i].gamma = 1;
    }

    return momenta;
}

std::vector<Momentum> generate_initial_electron_momenta_random_velocity(
    std::size_t num_electrons, uint32_t seed)
{
    std::vector<Momentum> momenta(num_electrons);

    constexpr Real max_velocity = (5.0 / 1000.0) * c;

    std::uniform_real_distribution<Real>
        unif_v(0.0, max_velocity);

    std::default_random_engine rng(seed);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        const Vector3D v{unif_v(rng), unif_v(rng), unif_v(rng)};

        const auto beta = v.norm() / c;

        momenta[i].vx = v.x;
        momenta[i].vy = v.y;
        momenta[i].vz = v.z;
        momenta[i].gamma = 1.0 / sqrt(1 - beta * beta);
    }

    return momenta;
}
