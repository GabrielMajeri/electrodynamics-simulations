module angular_momentum
   use iso_fortran_env, only: dp => real64
   use constants, only: num_particles, particle_mass

   implicit none (type, external)

contains

   subroutine compute_angular_momenta_in_z_direction(positions, momenta, angular_momenta)
      real(kind=dp), dimension(num_particles, 4), intent(in) :: &
         positions(:, :), momenta(:, :)

      real(kind=dp), dimension(num_particles), allocatable, intent(out) :: angular_momenta(:)

      integer :: i

      allocate(angular_momenta(1:num_particles))

      write(*, '("Computing angular momenta in the z direction")')

      do i = 1, num_particles
         ! Compute m * (x * v_y - y * v_x)
         angular_momenta(i) = particle_mass * (positions(i, 2) * momenta(i, 3) - positions(i, 3) * momenta(i, 2))
      end do

      write(*, '("Done computing angular momenta")')

   end subroutine compute_angular_momenta_in_z_direction

end module angular_momentum

