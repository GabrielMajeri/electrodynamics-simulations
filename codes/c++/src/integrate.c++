#include "integrate.h++"

#include <complex>
#include <iostream>

#include "beam.h++"
#include "common.h++"
#include "constants.h++"

using namespace std::complex_literals;

std::pair<Position, Momentum> perform_integration_step(Position previous_position, Momentum previous_momentum)
{
    const auto laboratory_time = previous_position.t;
    const auto position_vector = Vector3D::from_position(previous_position);

    // Compute EM field vectors for previous position
    auto [electric_field, magnetic_field] =
        laguerre_gauss_beam_electric_and_magnetic_field(position_vector, laboratory_time);

    const auto cf = cutoff(laboratory_time - previous_position.z / c, phi_0, tau_0);
    electric_field = cf * electric_field;
    magnetic_field = cf * magnetic_field;

    // Symplectic Euler integration step
    const auto acceleration = compute_acceleration(previous_momentum, electric_field, magnetic_field);

    const auto new_momentum = previous_momentum + integration_time_step * acceleration;
    const auto new_position = previous_position + integration_time_step * new_momentum;

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

    return std::make_pair(new_position, new_momentum);
}

Acceleration compute_acceleration(Momentum previous_momentum, Vector3D electric_field, Vector3D magnetic_field)
{
    const auto agamma = previous_momentum.vx * electric_field.x / c + previous_momentum.vy * electric_field.y / c + previous_momentum.vz * electric_field.z / c;
    const auto ax = previous_momentum.gamma * electric_field.x / c + previous_momentum.vy * magnetic_field.z - previous_momentum.vz * magnetic_field.y;
    const auto ay = previous_momentum.gamma * electric_field.y / c - previous_momentum.vx * magnetic_field.z + previous_momentum.vz * magnetic_field.x;
    const auto az = previous_momentum.gamma * electric_field.z / c + previous_momentum.vx * magnetic_field.y - previous_momentum.vy * magnetic_field.x;

    const Acceleration acceleration_direction{agamma, ax, ay, az};
    return charge_to_mass_ratio * acceleration_direction;
}
