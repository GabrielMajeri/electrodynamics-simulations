module integrate
   use iso_fortran_env, only: dp => real64
   use constants, only: c, num_particles, integration_duration, integration_time_step
   use vector
   use beam, only: compute_laguerre_gauss_beam_electric_and_magnetic_field

   implicit none (type, external)

contains

   subroutine integrate_trajectories(initial_positions, initial_momenta, final_positions, final_momenta)
      real(kind=dp), dimension(num_particles, 4), intent(in) :: &
         initial_positions(:, :), initial_momenta(:, :)

      real(kind=dp), dimension(num_particles, 4), allocatable, intent(out) :: &
         final_positions(:, :), final_momenta(:, :)

      real(kind=dp), dimension(num_particles, 4), allocatable :: positions(:, :), momenta(:, :)

      integer, parameter :: num_integration_steps = ceiling(integration_duration / integration_time_step)

      integer :: particle_index, step
      real(kind=dp) :: time

      integer(kind=selected_int_kind(18)) :: rate, start_time, end_time
      real(kind=dp) :: duration

      type(vec4_t) :: previous_position, previous_momentum, acceleration, &
         new_position, new_momentum
      type(vec3_t) :: position_vector, E, B

      allocate(positions(num_particles, 4))
      allocate(momenta(num_particles, 4))

      positions(:, :) = initial_positions
      momenta(:, :) = initial_momenta

      write(*, '("Starting to integrate trajectories")')

      call system_clock(count_rate=rate)

      call system_clock(start_time)

      !$omp parallel private(time, E, B)

      !$omp do
      do particle_index = 1, num_particles
         time = 0.0
         do step = 1, num_integration_steps
            previous_position = to_vec4(positions(particle_index, 1:4))
            previous_momentum = to_vec4(momenta(particle_index, 1:4))

            position_vector = vec3_t(previous_position%x, previous_position%y, previous_position%z)

            call compute_laguerre_gauss_beam_electric_and_magnetic_field(position_vector, time, E, B)

            acceleration = compute_acceleration(previous_momentum, E, B)

            new_momentum = previous_momentum + integration_time_step * acceleration
            new_position = previous_position + integration_time_step * new_momentum

            positions(particle_index, 1:4) = vec4_to_array(new_position)
            momenta(particle_index, 1:4) = vec4_to_array(new_momentum)

            time = time + integration_time_step
         end do
      end do
      !$omp end parallel

      call system_clock(end_time)

      duration = real(end_time - start_time, dp)/rate

      write(*, '("Done integrating trajectories")')
      write(*, '("Took ", F14.10, " seconds")') duration

      final_positions = positions
      final_momenta = momenta

   end subroutine integrate_trajectories

   function compute_acceleration(momentum, E, B) result(acc)
      type(vec4_t), intent(in) :: momentum
      type(vec3_t), intent(in) :: E, B

      type(vec4_t) :: p, acc

      p = momentum
      acc = vec4_t( &
         p%x * E%x / c + p%y * E%y/c + p%z * E%z/c, &
         p%a * E%x / c + p%y * B%z - p%z * B%y, &
         p%a * E%y / c - p%x * B%z + p%z * B%x, &
         p%a * E%z / c + p%x * B%y - p%y * B%x &
         )

   end function compute_acceleration

end module integrate
