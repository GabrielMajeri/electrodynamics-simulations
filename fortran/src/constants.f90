module constants
   use types, only : real_kind

   implicit none

   public

   !> Number of electrons
   integer, parameter :: num_particles = 64 * 1024

   !> Circle constant PI
   real(real_kind), parameter :: pi = 4 * atan(1.0)

   !> Speed of light in vacuum (in atomic units) -- basically the same as
   !> the fine structure constant in this case.
   real(real_kind), parameter :: c = 137.036

   !> Mass of the electron (in atomic units)
   real(real_kind), parameter :: particle_mass = 1.0

   !> Charge of the electron (in atomic units)
   real(real_kind), parameter :: particle_charge = -1.0

   !> q/m term which shows up in equations of motion
   real(real_kind), parameter :: charge_to_mass_ratio = particle_charge / particle_mass

   !> Radial index (parameter "p" in formulae)
   integer, parameter :: radial_index = 2

   !> Azimuthal index (parameter "l" in formulae)
   integer, parameter :: azimuthal_index = -2

   !> Angular velocity of laser beam
   real(real_kind), parameter :: omega = 0.057

   real(real_kind), parameter :: a_0 = 1e-2
   real(real_kind), parameter :: amplitude = a_0 * particle_mass * c * omega / abs(particle_charge)

   !> Wavelength of monochromatic laser light (in atomic units)
   real(real_kind), parameter :: lambda = 2 * pi * c / omega
   real(real_kind), parameter :: wavelength = lambda

   !> Wavenumber, k
   real(real_kind), parameter :: wavenumber = 2 * pi / lambda

   !> Radius at beam waist (z = 0)
   real(real_kind), parameter :: waist_radius = 75 * lambda

   !> Radius of disk in which initial conditions lie
   real(real_kind), parameter :: disk_radius = (1.75 + radial_index) * waist_radius

   !> Duration of the laser pulse (Gaussian time envelope)
   real(real_kind), parameter :: tau_0 = 10 / omega
   real(real_kind), parameter :: phi_0 = 3 * tau_0

   real(real_kind), parameter :: integration_start_time = 0.0
   real(real_kind), parameter :: integration_end_time = 6 * tau_0
   real(real_kind), parameter :: integration_time_step = 1e-1
   real(real_kind), parameter :: integration_duration = integration_end_time - integration_start_time

   complex(real_kind), parameter :: polarization_x = complex(1/sqrt(2.0_real_kind), 0)
   complex(real_kind), parameter :: polarization_y = complex(0, 1/sqrt(2.0_real_kind))

end module constants
