#include "angular_momentum.h++"

std::vector<Real> compute_angular_momenta_in_z_direction(
    Real particle_mass,
    std::vector<Position> positions, std::vector<Momentum> momenta)
{
    std::size_t num_particles = positions.size();
    std::vector<Real> angular_momenta(num_particles);

    for (std::size_t index = 0; index < num_particles; ++index)
    {
        const auto position = positions[index];
        const auto momentum = momenta[index];

        angular_momenta[index] = particle_mass * (position.x * momentum.vy - position.y * momentum.vx);
    }

    return angular_momenta;
}
