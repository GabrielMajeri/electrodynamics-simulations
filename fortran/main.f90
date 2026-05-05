program laguerre_gauss_beam_angular_momentum_transfer
   use :: iso_c_binding, only : c_double
   use initial_conditions_generation, only : initial_conditions, generate_initial_conditions

   implicit none (type, external)

   integer, parameter :: real_kind = c_double

   ! Number of electrons
   integer, parameter :: num_particles = 16 * 1024

   ! Radial index (parameter "p" in formula)
   integer, parameter :: radial_index = 2

   real(real_kind), parameter :: pi = 4 * atan(1.0)

   ! Speed of light in vacuum (in atomic units) -- basically the same as
   ! the fine structure constant in this case.
   real(real_kind), parameter :: c = 137.036

   ! Angular velocity of laser beam
   real(real_kind), parameter :: omega = 0.057

   ! Wavelength of monochromatic laser light (in atomic units)
   real(real_kind), parameter :: lambda = 2 * pi * c / omega

   ! Radius at beam waist (z = 0)
   real(real_kind), parameter :: waist_radius = 75 * lambda

   real(real_kind), parameter :: R_max = (1.75 + radial_index) * waist_radius

   type(initial_conditions(num_particles=num_particles)) :: initial_conds

   initial_conds = generate_initial_conditions(num_particles, R_max)

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
   ! - Write NumPy output code (or maybe use CSV export format...)

end program laguerre_gauss_beam_angular_momentum_transfer
