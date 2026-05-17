#include "integrate.h++"

#include <complex>
#include <iostream>

#include "common.h++"
#include "constants.h++"

using namespace std::complex_literals;

#ifdef _OPENMP
static void field_add(std::vector<ComplexVector3D> &inout, const std::vector<ComplexVector3D> &in)
{
    for (std::size_t i = 0; i < in.size(); ++i)
    {
        inout[i] += in[i];
    }
}
#pragma omp declare reduction(                                                 \
        field_add : std::vector<ComplexVector3D> : field_add(omp_out, omp_in)) \
    initializer(omp_priv = omp_orig)
#endif

IntegrationResult integrate_trajectories(
    std::vector<Position> initial_positions, std::vector<Momentum> initial_momenta)
{
    // Determine number of particles
    const std::size_t num_particles = initial_positions.size();
    assert(num_particles == initial_momenta.size());

    // Allocate some buffers to store positions/momenta during integration
    std::vector<Position> positions = initial_positions;
    std::vector<Momentum> momenta = initial_momenta;

    const Real integration_duration = integration_end_time - integration_start_time;
    const std::size_t num_steps = integration_duration / integration_time_step;

    std::vector<Vector3D> detector_positions(num_detector_points);
    constexpr auto detector_z = 1000 * lambda;

    for (std::size_t row = 0; row < detector_grid_size_y; ++row)
    {
        for (std::size_t column = 0; column < detector_grid_size_x; ++column)
        {
            const auto y = -detector_height + row * (2 * detector_height) / detector_grid_size_y;
            const auto x = -detector_width + column * (2 * detector_width) / detector_grid_size_x;
            detector_positions[row * detector_grid_size_x + column] = {x, y, detector_z};
        }
    }

    std::vector<ComplexVector3D> electric_field(num_detector_points), magnetic_field(num_detector_points);

    Position *positions_arr = positions.data();
    Momentum *momenta_arr = momenta.data();

#ifdef _OPENMP
#pragma omp parallel for reduction(field_add : electric_field)
#else
#ifdef _OPENACC
#pragma acc parallel loop copy(positions_arr[ : num_particles], momenta_arr[ : num_particles])
#endif
#endif
    for (std::size_t particle_index = 0; particle_index < num_particles; ++particle_index)
    {
        auto current_time = 0.0;
        for (std::size_t step = 0; step <= num_steps; ++step)
        {
            const auto previous_position = positions_arr[particle_index];
            const auto previous_momentum = momenta_arr[particle_index];

            const auto [new_position, new_momentum] = perform_integration_step(previous_position, previous_momentum);

            // \symfrak{R}_0 (for this particle)
            const auto initial_position = Vector3D::from_position(initial_positions[particle_index]);

            for (std::size_t detector_index = 0; detector_index < num_detector_points; ++detector_index)
            {
                const auto particle_position = Vector3D::from_position(new_position);
                const auto particle_velocity = Vector3D::from_momentum(new_momentum);

                // r_0(t) = r(t) - R_0
                const auto particle_displacement = particle_position - initial_position;

                // x_0(t) = x - R_0
                const auto detector_displacement = detector_positions[detector_index] - initial_position;

                // R(x_0, t) = x_0 - r_0(t) = (x - R_0) - (r(t) - R_0) = x - r(t)
                const auto displacement = detector_displacement - particle_displacement;
                const auto displacement_norm = displacement.norm();

                // n(x_0, t) = R(x_0, t)/|R(x_0, t)|
                const auto view_direction = displacement / displacement_norm;

                // exp(i * omega * (t + R(x_0, t)/c))
                const auto oscillatory_kernel = std::exp(1i * omega * (current_time + displacement_norm / c));

                // v/c
                const auto beta = particle_velocity / c;

                // O(1/|R|) term
                // ((i * omega) / c) * (beta(t) - n(x_0, t)) / |R(x_0, t)|
                const auto first_term = ((1i * omega) / c) * ComplexVector3D::from((beta - view_direction) / displacement_norm);

                // O(1/|R|^2) term
                // n(x_0, t) / |R(x_0, t)|^2
                const auto second_term = ComplexVector3D::from(view_direction / (displacement_norm * displacement_norm));

                // Riemann summation
                electric_field[detector_index] += integration_time_step * oscillatory_kernel * (first_term + second_term);
            }

            positions_arr[particle_index] = new_position;
            momenta_arr[particle_index] = new_momentum;

            current_time += integration_time_step;
        }
    }

    return {
        positions,
        momenta,
        detector_positions,
        electric_field,
        magnetic_field,
    };
}

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
