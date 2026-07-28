#include "integrate.h++"

#include <complex>
#include <iostream>

#include "beam.h++"
#include "common.h++"
#include "constants.h++"

using namespace std::complex_literals;

static void check_integration_results(Momentum momentum, Acceleration acceleration)
{
    if (momentum.u0 < c - error_tolerance)
    {
        std::cout << "Lorentz factor dropped below unity: " << momentum.u0 << std::endl;
        std::exit(1);
    }

    const auto inner_product = acceleration.du1 * momentum.u1 + acceleration.du2 * momentum.u2 + acceleration.du3 * momentum.u3 - acceleration.du0 * momentum.u0;

    if (std::abs(inner_product) > error_tolerance)
    {
        std::cout << "Inner product is non-zero: " << std::abs(inner_product) << std::endl;
        std::exit(1);
    }
}

std::pair<Position, Momentum> perform_integration_step_euler(Position previous_position, Momentum previous_momentum)
{
    const auto laboratory_time = previous_position.t / c;
    const auto position_vector = Vector3D::from_position(previous_position);

    // Compute EM field vectors for previous position
    // auto [electric_field, magnetic_field] = plane_wave_electric_and_magnetic_field(position_vector, laboratory_time);
    // auto [electric_field, magnetic_field] = gauss_beam_electric_and_magnetic_field(position_vector, laboratory_time);
    auto [electric_field, magnetic_field] =
        laguerre_gauss_beam_electric_and_magnetic_field(position_vector, laboratory_time);

    const auto cf = cutoff((previous_position.t - previous_position.z) / c, phi_0, tau_0);
    electric_field = cf * electric_field;
    magnetic_field = cf * magnetic_field;

    // Symplectic Euler integration step
    const auto acceleration = compute_acceleration(previous_momentum, electric_field, magnetic_field);

    if (check_for_errors)
    {
        check_integration_results(previous_momentum, acceleration);
    }

    const auto new_momentum = previous_momentum + integration_time_step * acceleration;
    const auto new_position = previous_position + integration_time_step * new_momentum;

    return std::make_pair(new_position, new_momentum);
}

static Acceleration compute_intermediate_acceleration(Real time, Vector3D position, Momentum momentum)
{
    // auto [electric_field, magnetic_field] = plane_wave_electric_and_magnetic_field(position, time);
    // auto [electric_field, magnetic_field] = gauss_beam_electric_and_magnetic_field(position, time);
    auto [electric_field, magnetic_field] =
        laguerre_gauss_beam_electric_and_magnetic_field(position, time);

    const auto cf = cutoff(time - position.z / c, phi_0, tau_0);
    electric_field = cf * electric_field;
    magnetic_field = cf * magnetic_field;

    return compute_acceleration(momentum, electric_field, magnetic_field);
}

std::pair<Position, Momentum> perform_integration_step_rk4(
    Position previous_position, Momentum previous_momentum)
{
    const auto laboratory_time = previous_position.t / c;
    const auto position_vector = Vector3D::from_position(previous_position);

    const auto k_1 = compute_intermediate_acceleration(laboratory_time, position_vector, previous_momentum);
    const auto k_2 = compute_intermediate_acceleration(
        laboratory_time + integration_time_step / 2,
        position_vector,
        previous_momentum + integration_time_step / 2 * k_1);
    const auto k_3 = compute_intermediate_acceleration(
        laboratory_time + integration_time_step / 2,
        position_vector,
        previous_momentum + integration_time_step / 2 * k_2);
    const auto k_4 = compute_intermediate_acceleration(
        laboratory_time + integration_time_step,
        position_vector,
        previous_momentum + integration_time_step * k_3);

    if (check_for_errors)
    {
        check_integration_results(previous_momentum, k_1);
    }

    const auto acceleration = (1 / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4);

    const auto new_momentum = previous_momentum + integration_time_step * acceleration;
    const auto new_position = previous_position + integration_time_step * new_momentum;

    return std::make_pair(new_position, new_momentum);
}

Acceleration compute_acceleration(Momentum previous_momentum, Vector3D electric_field, Vector3D magnetic_field)
{
    const auto du0 = previous_momentum.u1 * electric_field.x / c + previous_momentum.u2 * electric_field.y / c + previous_momentum.u3 * electric_field.z / c;
    const auto du1 = previous_momentum.u0 * electric_field.x / c + previous_momentum.u2 * magnetic_field.z - previous_momentum.u3 * magnetic_field.y;
    const auto du2 = previous_momentum.u0 * electric_field.y / c - previous_momentum.u1 * magnetic_field.z + previous_momentum.u3 * magnetic_field.x;
    const auto du3 = previous_momentum.u0 * electric_field.z / c + previous_momentum.u1 * magnetic_field.y - previous_momentum.u2 * magnetic_field.x;

    const Acceleration acceleration_direction{du0, du1, du2, du3};
    return charge_to_mass_ratio * acceleration_direction;
}

void integrate_scattered_field(
    Real current_time,
    const Position &position, const Momentum &momentum,
    const Position &initial_position,
    const std::vector<Vector3D> &detector_positions,
    std::vector<ComplexVector3D> &electric_field,
    std::vector<ComplexVector3D> &magnetic_field)
{
    // \symfrak{R}_0 (for this particle)
    const auto initial_position_vector = Vector3D::from_position(initial_position);

    const auto num_detector_points = detector_positions.size();
    for (std::size_t detector_index = 0; detector_index < num_detector_points; ++detector_index)
    {
        const auto particle_position = Vector3D::from_position(position);
        const auto particle_velocity = Vector3D::from_momentum(momentum);

        // r_0(t) = r(t) - R_0
        const auto particle_displacement = particle_position - initial_position_vector;

        // x_0(t) = x - R_0
        const auto detector_displacement = detector_positions[detector_index] - initial_position_vector;

        // R(x_0, t) = x_0 - r_0(t) = (x - R_0) - (r(t) - R_0) = x - r(t)
        const auto displacement = detector_displacement - particle_displacement;
        const auto displacement_norm = displacement.norm();

        // n(x_0, t) = R(x_0, t)/|R(x_0, t)|
        const auto view_direction = displacement / displacement_norm;

        // exp(i * omega * (t + R(x_0, t)/c))
        const auto oscillatory_kernel = std::exp(1i * omega * (current_time + displacement_norm / c));

        // v/c
        const auto beta = particle_velocity / c;

        //===== Electric field terms =====
        // Common term: n(x_0, t) \times (n(x_0, t) \times \beta(t))
        const auto electric_field_common_term = view_direction.cross(view_direction.cross(beta));

        // O(1/|R|) term
        // - ((i * omega) / c) * (common term) / |R(x_0, t)|
        const auto electric_field_first_term = -((1i * omega) / c) * ComplexVector3D::from(electric_field_common_term / displacement_norm);

        const auto displacement_norm_squared = displacement_norm * displacement_norm;

        // O(1/|R|^2) term
        // [(common term) + n(x_0, t) * (1 + dot(n(x_0, t), \beta(t)))] / |R(x_0, t)|^2
        const auto electric_field_second_term = ComplexVector3D::from((electric_field_common_term + view_direction * (1 + view_direction.dot(beta))) / displacement_norm_squared);

        const auto n_cross_beta = view_direction.cross(beta);

        //===== Magnetic field terms =====
        // O(1/|R|) term
        const auto magnetic_field_first_term = ((1i * omega) / c) * ComplexVector3D::from(n_cross_beta / displacement_norm);

        // O(1/|R|^2) term
        const auto magnetic_field_second_term = ComplexVector3D::from(n_cross_beta / displacement_norm_squared);

        // Riemann summation
        electric_field[detector_index] += integration_time_step * oscillatory_kernel * (electric_field_first_term + electric_field_second_term);
        magnetic_field[detector_index] += integration_time_step * oscillatory_kernel * (magnetic_field_first_term - magnetic_field_second_term);
    }
}
