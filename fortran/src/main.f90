program laguerre_gauss_beam_angular_momentum_transfer
   use constants, only : num_particles
   use types, only : real_kind
   use initial_conditions_generation, only : generate_initial_conditions
   use numpy, only : write_to_npy_file
   use integrate, only : integrate_trajectories
   use angular_momentum, only : compute_angular_momenta_in_z_direction

#ifdef _OPENMP
   use omp_lib, only: omp_get_num_procs
#endif

   implicit none (type, external)

   real(real_kind), dimension(4, num_particles), allocatable :: &
      initial_positions(:, :), initial_momenta(:, :), &
      final_positions(:, :), final_momenta(:, :)

   real(real_kind), dimension(num_particles), allocatable :: angular_momenta(:)

#ifdef _OPENMP
   write (*, '("Using OpenMP on up to ", I0, " processors")') omp_get_num_procs()
#endif

   ! Generate initial positions and initial momenta
   call generate_initial_conditions(initial_positions, initial_momenta)

   ! Save the initial electron positions to file
   call write_to_npy_file("initial_positions.npy", initial_positions)

   ! Integrate trajectories for a fixed amount of time
   call integrate_trajectories(initial_positions, initial_momenta, final_positions, final_momenta)

   ! Compute the angular momentum of each particle
   call compute_angular_momenta_in_z_direction(final_positions, final_momenta, angular_momenta)

   ! Save final angular momenta to disk
   call write_to_npy_file("angular_momenta.npy", reshape(angular_momenta, [1, num_particles]))

   write(*, '("Done")')

end program laguerre_gauss_beam_angular_momentum_transfer
