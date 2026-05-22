program laguerre_gauss_beam_angular_momentum_transfer
   use constants, only : num_detector_points, num_particles
   use types, only : real_kind
   use initial_conditions_generation, only : generate_initial_conditions
   use detector, only : initialize_detector_positions
   use numpy, only : write_to_npy_file
   use integrate, only : simulate_analytic_trajectories, integrate_trajectories
   use angular_momentum, only : compute_angular_momenta_in_z_direction

   use omp_lib, only: omp_get_num_procs

   implicit none (type, external)

   real(real_kind), dimension(4, num_particles), allocatable :: &
      initial_positions(:, :), initial_momenta(:, :), &
      final_positions(:, :), final_momenta(:, :)

   real(real_kind), dimension(3, num_detector_points), allocatable :: detector_positions(:, :)

   complex(real_kind), dimension(3, num_detector_points), allocatable :: electric_field(:, :)

   real(real_kind), dimension(num_particles), allocatable :: angular_momenta(:)

   write (*, '("Using OpenMP on up to ", I0, " processors")') omp_get_num_procs()

   ! Generate initial positions and initial momenta
   call generate_initial_conditions(initial_positions, initial_momenta)

   ! Save the initial electron positions to file
   call write_to_npy_file("initial_positions.npy", initial_positions)

   ! Initialize the detector positions vector
   call initialize_detector_positions(detector_positions)

   ! Save detector positions to disk
   call write_to_npy_file("detector_positions.npy", detector_positions)

   ! call simulate_analytic_trajectories(initial_positions, initial_momenta, detector_positions, &
   ! final_positions, final_momenta, electric_field)

   ! Integrate trajectories for a fixed amount of time
   call integrate_trajectories(initial_positions, initial_momenta, detector_positions, &
      final_positions, final_momenta, electric_field)

   ! Save electric field at the detector to disk
   call write_to_npy_file("electric_field.npy", electric_field)

   ! Compute the angular momentum of each particle
   call compute_angular_momenta_in_z_direction(final_positions, final_momenta, angular_momenta)

   ! Save final angular momenta to disk
   call write_to_npy_file("angular_momenta.npy", angular_momenta)

   deallocate(final_positions)
   deallocate(final_momenta)
   deallocate(electric_field)

   write(*, '("Done")')

end program laguerre_gauss_beam_angular_momentum_transfer
