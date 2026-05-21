module initial_conditions_generation
   use iso_fortran_env, only: dp => real64
   use constants, only: disk_radius, large_circle_radius, num_particles, pi

   implicit none (type, external)

   integer, parameter :: k = kind(0.0_dp)

contains

   subroutine generate_initial_conditions(initial_positions, initial_momenta)
      real(kind=k), dimension(4, num_particles), allocatable, intent(out) :: &
         initial_positions(:, :), initial_momenta(:, :)

      call generate_positions_on_circle(initial_positions)
      ! call generate_random_positions_within_disk(initial_positions)

      call initialize_momenta(initial_momenta)

   end subroutine generate_initial_conditions

   subroutine generate_positions_on_circle(positions)
      real(kind=k), dimension(4, num_particles), allocatable, intent(out) :: &
         positions(:, :)

      integer :: particle_index
      real(kind=k) :: f

      allocate(positions(4, num_particles))

      do particle_index = 1, num_particles
         f = real(particle_index - 1, kind=dp) / real(num_particles, kind=dp)
         positions(:, particle_index) = [ &
            0.0_dp, &
            large_circle_radius * cos(2 * pi * f), &
            large_circle_radius * sin(2 * pi * f), &
            0.0_dp &
            ]
      end do

   end subroutine generate_positions_on_circle

   subroutine generate_random_positions_within_disk(positions)
      real(kind=k), dimension(4, num_particles), allocatable, intent(out) :: &
         positions(:, :)

      integer, allocatable :: seed(:)
      integer :: n, i

      real(kind=k) :: u(2), disk_radius_squared, radius, theta
      real(kind=k) :: start_time, end_time, duration

      ! Seed the random number generator

      ! Determine how long the state vector should be
      call random_seed(size=n)

      write(*, '("Random number generator seed state vector is ", I0, " integers long")') n

      allocate(seed(n))
      do i = 1, n
         ! Fix a predictable seed
         seed(i) = i
      end do

      call random_seed(put=seed)
      write(*, '("Successfully seeded RNG")')

      allocate(positions(4, num_particles), source=0.0_dp)

      disk_radius_squared = disk_radius * disk_radius

      call cpu_time(start_time)

      do i = 1, num_particles
         call random_number(u)
         radius = sqrt(u(1) * disk_radius_squared)
         theta = 2 * pi * u(2)

         positions(2, i) = radius * cos(theta)
         positions(3, i) = radius * sin(theta)
      end do

      call cpu_time(end_time)

      duration = end_time - start_time

      write(*, '("Done generating initial conditions")')
      write(*, '("Took ", F8.6, " seconds")') duration

   end subroutine generate_random_positions_within_disk

   pure subroutine initialize_momenta(momenta)
      real(kind=k), dimension(4, num_particles), allocatable, intent(out) :: &
         momenta(:, :)

      integer :: particle_index

      allocate(momenta(4, num_particles), source=0.0_dp)

      do particle_index = 1, num_particles
         momenta(1, particle_index) = 1
      end do

   end subroutine initialize_momenta

end module initial_conditions_generation
