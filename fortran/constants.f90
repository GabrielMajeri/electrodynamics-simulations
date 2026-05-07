module constants
   use types, only : real_kind

   implicit none

   public

   ! Number of electrons
   integer, parameter :: num_particles = 16 * 1024

   ! Radial index (parameter "p" in formula)
   integer, parameter :: radial_index = 2

   ! Circle constant PI
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

   ! Radius of disk in which initial conditions lie
   real(real_kind), parameter :: R_max = (1.75 + radial_index) * waist_radius

end module constants
