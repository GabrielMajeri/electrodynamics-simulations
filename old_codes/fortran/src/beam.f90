module beam
   use iso_fortran_env, only: dp => real64
   use constants, only: amplitude, azimuthal_index, c, lambda, omega, &
      pi, polarization_x, polarization_y, &
      radial_index, waist_radius, wavenumber
   use vector

   implicit none (type, external)

   real (kind=dp), parameter :: tolerance = 1e-5

contains

   pure subroutine compute_laguerre_gauss_beam_electric_and_magnetic_field(position, time, E, B)
      type(vec3_t), intent(in) :: position
      real(kind=dp), intent(in) :: time
      type(vec3_t), intent(out) :: E, B

      !> z_R
      real(kind=dp), parameter :: rayleigh_length = pi * (waist_radius**2) / lambda

      real(kind=dp) :: x, y, z, rho, phi, width, &
         r_over_width, r_over_width_squared, &
         radius_of_curvature, curvature, &
         gouy_phase, magnitude
      complex(kind=dp) :: phase, coefficient, &
         E_x, E_y, E_z, &
         B_x, B_y, B_z
      integer :: abs_l

      x = position%x
      y = position%y
      z = position%z

      rho = sqrt(x**2 + y**2)
      phi = atan2(y, x)

      !> w(z)
      width = waist_radius * sqrt(1 + (z / rayleigh_length) ** 2)

      !> r / w(z)
      r_over_width = rho / width
      r_over_width_squared = r_over_width ** 2

      !> |l|
      abs_l = abs(azimuthal_index)

      !> R(z)
      if (abs(z) < tolerance) then
         radius_of_curvature = 0
      else
         radius_of_curvature = z * (1 + (rayleigh_length / position%z) ** 2)
      end if

      !> r^2/(2 * R(z))
      if (abs(radius_of_curvature) < tolerance) then
         curvature = 0
      else
         curvature = (rho ** 2) / (2 * radius_of_curvature)
      end if

      !> \psi(z)
      gouy_phase = atan2(z, rayleigh_length)

      magnitude = amplitude * (waist_radius / width) &
         * ((sqrt(2.0_dp) * r_over_width) ** abs_l) &
         * laguerre_polynomial(radial_index, real(abs_l, kind=dp), 2 * r_over_width_squared) &
         * exp(-r_over_width_squared)

      phase = exp(complex(0, 1) * ( &
         omega * time &
         - (wavenumber * z + wavenumber * curvature + azimuthal_index * phi &
         - (2 * radial_index + abs_l + 1) * gouy_phase)&
         ))

      coefficient = magnitude * phase

      E_x = coefficient * polarization_x
      E_y = coefficient * polarization_y
      E_z = complex(0, 2) / (wavenumber * (width ** 2)) * (x * E_x + y * E_y)

      E = vec3_t(E_x%re, E_y%re, E_z%re)

      B_x = -E_y / c
      B_y = E_x / c
      B_z = complex(0, 1) / (omega * (width ** 2)) * (y * E_x - x * E_y)

      B = vec3_t(B_x%re, B_y%re, B_z%re)

   end subroutine

   pure recursive function laguerre_polynomial(n, alpha, x) result(f)
      integer, intent(in) :: n
      real(kind=dp), intent(in) :: alpha
      real(kind=dp), intent(in) :: x
      real(kind=dp) :: f

      if (n == 0) then
         f = 1
      else if (n == 1) then
         f = 1 + alpha - x
      else if (n == 2) then
         f = 0.5 * (x ** 2 - 2 * (alpha + 2) * x + (alpha + 1) * (alpha + 2))
      else
         f = ((2 * n - 1 + alpha - x) * laguerre_polynomial(n - 1, alpha, x) &
            - (n - 1 + alpha) * laguerre_polynomial(n - 2, alpha, x)) / n
      end if

   end function laguerre_polynomial

end module beam
