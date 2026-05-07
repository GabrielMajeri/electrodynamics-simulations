program laguerre_gauss_beam_angular_momentum_transfer
   use types, only : real_kind
   use constants, only : num_particles, R_max
   use initial_conditions_generation, only : generate_initial_conditions
   use numpy, only : write_to_npy_file

   implicit none (type, external)

   real(real_kind), dimension(num_particles, 4), allocatable :: initial_positions(:, :), initial_momenta(:, :)

   ! Generate initial positions and initial momenta
   call generate_initial_conditions(num_particles, R_max, initial_positions, initial_momenta)

   ! Save the initial electron positions to file
   call write_to_npy_file("initial_positions.npy", initial_positions)

   ! TODO:
   ! - Start writing numerical integration loop
   !
   !   - Perform loop unrolling (if possible)
   !
   !   - Write function to evaluate Laguerre-Gauss beam
   !     - Write function to compute Laguerre polynomial
   !
   !   - Compute matrix-vector product between EM tensor and 4-momentum
   !
   ! - Write NumPy output code

end program laguerre_gauss_beam_angular_momentum_transfer
