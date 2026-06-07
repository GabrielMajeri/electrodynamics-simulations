#include "trajectory.h++"

#include <cassert>

#include "beam.h++"
#include "constants.h++"
#include "integrate.h++"

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

IntegrationResult analytic_trajectories(
    std::vector<Position> initial_positions,
    std::vector<Vector3D> detector_positions)
{
    // Determine number of particles
    const std::size_t num_particles = initial_positions.size();

    auto positions = initial_positions;
    std::vector<Momentum> momenta(num_particles);

    std::vector<Position> particle_trajectory(num_steps + 1);

    const auto num_detector_points = detector_positions.size();
    std::vector<ComplexVector3D> electric_field(num_detector_points), magnetic_field(num_detector_points);

#ifdef _OPENMP
#pragma omp parallel for reduction(field_add : electric_field)
#endif
    for (std::size_t particle_index = 0; particle_index < num_particles; ++particle_index)
    {
        const auto trajectory_center_x = large_circle_radius * std::cos((2 * pi * particle_index) / num_particles);
        const auto trajectory_center_y = large_circle_radius * std::sin((2 * pi * particle_index) / num_particles);

        const auto initial_position = initial_positions[particle_index];

        constexpr std::size_t num_wraps = 1;
        const auto initial_phase = (num_wraps * (2 * pi) * particle_index) / num_particles;
        // const auto initial_phase = std::atan2(initial_position.y, initial_position.x);
        const auto trajectory_amplitude = (0.1 / (2 * pi)) * lambda;

        auto current_time = 0.0;
        for (std::size_t step = 0; step <= num_steps; ++step)
        {
            const auto cf = cutoff(current_time, phi_0, tau_0);

            const auto new_position = Position{
                0.0,
                trajectory_center_x + cf * trajectory_amplitude * std::cos(omega * current_time - initial_phase),
                trajectory_center_y + cf * trajectory_amplitude * std::sin(omega * current_time - initial_phase),
                0,
            };
            const auto new_momentum = Momentum{
                0.0,
                -cf * trajectory_amplitude * omega * std::sin(omega * current_time - initial_phase),
                cf * trajectory_amplitude * omega * std::cos(omega * current_time - initial_phase),
                0,
            };

            integrate_scattered_field(
                current_time, new_position, new_momentum,
                initial_position, detector_positions,
                electric_field, magnetic_field);

            if (particle_index == 0)
            {
                particle_trajectory[step] = new_position;
            }

            positions[particle_index] = new_position;
            momenta[particle_index] = new_momentum;

            current_time += integration_time_step;
        }
    }

    return {
        positions,
        momenta,
        particle_trajectory,
        electric_field,
        magnetic_field,
    };
}

IntegrationResult integrate_trajectories(
    std::vector<Position> initial_positions,
    std::vector<Momentum> initial_momenta,
    std::vector<Vector3D> detector_positions)
{
    // Determine number of particles
    const std::size_t num_particles = initial_positions.size();
    assert(num_particles == initial_momenta.size());

    // Allocate some buffers to store positions/momenta during integration
    std::vector<Position> positions = initial_positions;
    std::vector<Momentum> momenta = initial_momenta;

    std::vector<Position> particle_trajectory(num_steps + 1);

    const auto num_detector_points = detector_positions.size();
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

            integrate_scattered_field(
                current_time, new_position, new_momentum,
                initial_positions[particle_index], detector_positions,
                electric_field, magnetic_field);

            if (particle_index == 0)
            {
                particle_trajectory[step] = new_position;
            }

            positions_arr[particle_index] = new_position;
            momenta_arr[particle_index] = new_momentum;

            current_time += integration_time_step;
        }
    }

    return {
        positions,
        momenta,
        particle_trajectory,
        electric_field,
        magnetic_field,
    };
}
