#include "integrate.h++"

#include <iostream>

#include "common.h++"
#include "constants.h++"

std::pair<std::vector<Position>, std::vector<Momentum>> integrate_trajectories(
    LaguerreGaussBeamParameters parameters, Real charge_to_mass_ratio,
    Real phi_0, Real tau_0,
    std::vector<Position> initial_positions, std::vector<Momentum> initial_momenta,
    Real integration_start_time, Real integration_end_time,
    Real time_step)
{
    // Determine number of particles
    const size_t num_particles = initial_positions.size();
    assert(num_particles == initial_momenta.size());

    // Allocate some buffers to store positions/momenta during integration
    std::vector<Position> positions = initial_positions;
    std::vector<Momentum> momenta = initial_momenta;

    const Real integration_duration = integration_end_time - integration_start_time;
    const size_t num_steps = integration_duration / time_step;

    Position *positions_arr = positions.data();
    Momentum *momenta_arr = momenta.data();

#ifdef _OPENMP
#pragma omp parallel for
#else
#ifdef _OPENACC
#pragma acc parallel loop copy(positions_arr[ : num_particles], momenta_arr[ : num_particles])
#endif
#endif
    for (size_t index = 0; index < num_particles; ++index)
    {
        Real current_time = 0;

        for (size_t step = 0; step <= num_steps; ++step)
        {
            const auto previous_position = positions_arr[index];
            const auto laboratory_time = previous_position.t;
            const auto position_vector = Vector3D::from_position(previous_position);

            // Compute EM field vectors for previous position
            auto [electric_field, magnetic_field] =
                laguerre_gauss_beam_electric_and_magnetic_field(parameters, position_vector, laboratory_time);

            const auto cf = cutoff(laboratory_time - previous_position.z / c, phi_0, tau_0);
            electric_field = cf * electric_field;
            magnetic_field = cf * magnetic_field;

            const auto previous_momentum = momenta_arr[index];

            // Symplectic Euler integration step
            const auto acceleration = compute_acceleration(previous_momentum, charge_to_mass_ratio, electric_field, magnetic_field);

            const auto new_momentum = previous_momentum + time_step * acceleration;
            const auto new_position = previous_position + time_step * new_momentum;

            if (check_for_errors)
            {
                if (new_momentum.gamma < 1 - error_tolerance)
                {
                    std::cout << "Lorentz factor dropped below unity: " << new_momentum.gamma << std::endl;
                    std::exit(1);
                }

                const auto inner_product = acceleration.dvx * previous_momentum.vx + acceleration.dvy * previous_momentum.vy + acceleration.dvz * previous_momentum.vz - acceleration.dgamma * previous_momentum.gamma;

                if (std::abs(inner_product) > error_tolerance)
                {
                    std::cout << "Inner product is non-zero: " << std::abs(inner_product) << std::endl;
                    std::exit(1);
                }
            }

            positions_arr[index] = new_position;
            momenta_arr[index] = new_momentum;
        }

        current_time += time_step;
    }

    return std::make_pair(positions, momenta);
}

Acceleration compute_acceleration(Momentum previous_momentum, Real charge_to_mass_ratio, Vector3D electric_field, Vector3D magnetic_field)
{
    const auto agamma = previous_momentum.vx * electric_field.x / c + previous_momentum.vy * electric_field.y / c + previous_momentum.vz * electric_field.z / c;
    const auto ax = previous_momentum.gamma * electric_field.x / c + previous_momentum.vy * magnetic_field.z - previous_momentum.vz * magnetic_field.y;
    const auto ay = previous_momentum.gamma * electric_field.y / c - previous_momentum.vx * magnetic_field.z + previous_momentum.vz * magnetic_field.x;
    const auto az = previous_momentum.gamma * electric_field.z / c + previous_momentum.vx * magnetic_field.y - previous_momentum.vy * magnetic_field.x;

    const Acceleration acceleration_direction{agamma, ax, ay, az};
    return charge_to_mass_ratio * acceleration_direction;
}
