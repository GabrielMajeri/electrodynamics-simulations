#include <chrono>
#include <filesystem>
#include <iostream>

#include "angular_momentum.h++"
#include "common.h++"
#include "constants.h++"
#include "initial_conditions.h++"
#include "npy_io.h++"
#include "detector.h++"
#include "trajectory.h++"
#include "types.h++"

int main()
{
    std::cout << "Starting Laguerre-Gauss beam angular momentum transfer simulation code" << std::endl;

    std::filesystem::create_directory("outputs");

#ifdef _OPENMP
    std::cout
        << "Using OpenMP with up to " << omp_get_num_procs() << " cores" << std::endl;
#else
#ifdef _OPENACC
    std::cout
        << "Using OpenACC\n";

    const auto device_type = acc_get_device_type();
    std::cout << "OpenACC device type: ";
    if (device_type == acc_device_host)
    {
        std::cout << "Host (CPU)";
    }
    else if (device_type == acc_device_nvidia)
    {
        std::cout << "NVIDIA";
    }
    else
    {
        std::cout << device_type;
    }
    std::cout << std::endl;

#else
    std::cout
        << "Warning: not using any sort of accelerator or parallelization" << std::endl;
#endif
#endif

    constexpr uint32_t seed = 42;

    std::cout << "Generating initial positions for " << num_electrons << " electrons, uniformly distributed within a disk of radius " << disk_radius << " in the x-y plane, centered at the origin" << std::endl;

    auto start = std::chrono::steady_clock::now();

    // const std::vector<Position> initial_electron_positions = generate_initial_electron_positions_on_circle(num_electrons, disk_radius);
    const std::vector<Position> initial_electron_positions = generate_initial_electron_positions_within_disk(num_electrons, disk_radius, seed);

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = finish - start;

    std::cout << "Generated " << num_electrons << " initial positions in " << elapsed_seconds.count() << " seconds" << std::endl;

    std::cout << "Writing initial electron positions to disk..." << std::endl;
    write_npy_file("outputs/initial_positions.npy", initial_electron_positions);

    const auto initial_electron_momenta = generate_initial_electron_momenta_stationary(num_electrons);
    // TODO(BUG): gamma drops below 1 when the electrons have some initial momenta
    // const auto initial_electron_momenta = generate_initial_electron_momenta_random_velocity(num_electrons, seed);

    std::cout << "Generating detector positions" << std::endl;
    const auto detector_positions = initialize_detector_positions();

    std::cout << "Saving detector positions to disk..." << std::endl;
    write_npy_file("outputs/detector_positions.npy", detector_positions);

    std::cout << "Integrating equations of motions from t_0 = " << integration_start_time
              << " up to t_final = " << integration_end_time
              << ", with a time step of " << integration_time_step << std::endl;

    start = std::chrono::steady_clock::now();

    // const auto integration_result = analytic_trajectories(initial_electron_positions, detector_positions);
    const auto integration_result = integrate_trajectories(initial_electron_positions, initial_electron_momenta, detector_positions);

    finish = std::chrono::steady_clock::now();
    elapsed_seconds = finish - start;

    std::cout << "Integrating " << num_electrons << " trajectories took " << elapsed_seconds.count() << " seconds" << std::endl;

#ifdef _OPENACC
    // BUGFIX: if I don't shutdown OpenACC explicitly here, it crashes (returns a non-zero exit code) at program exit
    acc_shutdown(acc_get_device_type());
#endif

    std::cout << "Writing sample particle trajectory to disk..." << std::endl;
    write_npy_file("outputs/particle_trajectory.npy", integration_result.particle_trajectory);
    write_npy_file("outputs/particle_momenta.npy", integration_result.particle_momenta);

    std::cout << "Writing emitted electric and magnetic fields to disk..." << std::endl;
    write_npy_file("outputs/electric_field.npy", integration_result.electric_field);
    write_npy_file("outputs/magnetic_field.npy", integration_result.magnetic_field);

    std::cout << "Computing angular momentum in the z direction for electrons in the final state" << std::endl;

    start = std::chrono::steady_clock::now();

    const auto angular_momenta = compute_angular_momenta_in_z_direction(
        m_e, integration_result.positions, integration_result.momenta);

    finish = std::chrono::steady_clock::now();
    elapsed_seconds = finish - start;

    std::cout << "Computing angular momenta for " << num_electrons << " electrons took " << elapsed_seconds.count() << " seconds" << std::endl;

    std::cout << "Writing angular momenta to disk..." << std::endl;
    write_npy_file("outputs/angular_momenta.npy", angular_momenta);

    std::cout << "Done" << std::endl;

    return 0;
}
