module initial_conditions_generation
   use iso_fortran_env, only: dp => real64

   implicit none

   integer, parameter :: k = kind(0.0_dp)

   type :: initial_conditions(num_particles)
      integer, len :: num_particles

      real(kind=k), dimension(num_particles, 4), allocatable :: initial_positions(:, :)
      real(kind=k), dimension(num_particles, 4), allocatable :: initial_momenta(:, :)
   end type initial_conditions

contains
   function generate_initial_conditions(num_particles, disk_radius) result(initial_conds)
      integer, intent(in) :: num_particles
      real(kind=k), intent(in) :: disk_radius
      type(initial_conditions(num_particles=num_particles)) :: initial_conds

      integer, allocatable :: seed(:)
      integer :: n, i

      real(kind=k), dimension(num_particles, 4), allocatable :: initial_positions(:, :)
      real(kind=k), dimension(num_particles, 4), allocatable :: initial_momenta(:, :)
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

      allocate(initial_positions(num_particles, 4), source=0.0_dp)
      allocate(initial_momenta(num_particles, 4), source=0.0_dp)

      disk_radius_squared = disk_radius * disk_radius

      call cpu_time(start_time)

      do i = 1, num_particles
         call random_number(u)
         radius = sqrt(u(1) * disk_radius_squared)
         theta = u(2)

         initial_positions(i, 2) = radius * cos(theta)
         initial_positions(i, 3) = radius * sin(theta)

         initial_momenta(i, 1) = 1
      end do

      call cpu_time(end_time)

      duration = end_time - start_time

      write(*, '("Done generating initial conditions")')
      write(*, '("Took ", F8.6, " seconds")') duration

   end function generate_initial_conditions

end module initial_conditions_generation
