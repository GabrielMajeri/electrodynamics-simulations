module detector
   use iso_fortran_env, only: dp => real64
   use constants, only: detector_width, lambda, num_detector_points

   implicit none (type, external)

contains

   pure subroutine initialize_detector_positions(detector_positions)
      real(kind=dp), allocatable, intent(out) :: detector_positions(:, :)

      real(kind=dp) :: x, y, z
      integer :: detector_index

      allocate(detector_positions(1:3, 1:num_detector_points))

      y = 0
      z = 1000 * lambda

      do detector_index = 1, num_detector_points
         x = - detector_width + (real(detector_index, kind=dp) / num_detector_points) * (2 * detector_width)

         detector_positions(:, detector_index) = [x, y, z]
      end do

   end subroutine initialize_detector_positions

end module detector
