program laguerre_gauss_beam_angular_momentum_transfer
   use :: iso_c_binding, only : c_double

   implicit none (type, external)

   integer, parameter :: real_kind = c_double

   ! Number of electrons
   integer, parameter :: num_particles = 2 * 8192

   ! Radial index (parameter "p" in formula)
   integer, parameter :: radial_index = 2

   real(real_kind), parameter :: pi = 4 * atan(1_real_kind)

   ! Speed of light in vacuum (in atomic units) -- basically the same as
   ! the fine structure constant in this case.
   real, parameter :: c = 137.036

   ! Angular velocity of laser beam
   real, parameter :: omega = 0.057

   ! Wavelength of monochromatic laser light (in atomic units)
   real, parameter :: lambda = 2 * pi * c / omega

   ! Radius at beam waist (z = 0)
   real, parameter :: waist_radius = 75 * lambda

   R_max = (1.75 + radial_index) * waist_radius

   ! This is a comment; it is ignored by the compiler.
   print '(a)', 'Hello, world!'
end program laguerre_gauss_beam_angular_momentum_transfer
