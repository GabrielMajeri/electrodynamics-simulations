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

std::vector<Momentum> generate_initial_electron_momenta(size_t num_electrons)
{
    std::vector<Momentum> momenta(num_electrons);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        momenta[i].gamma = 1;
    }

    return momenta;
}
