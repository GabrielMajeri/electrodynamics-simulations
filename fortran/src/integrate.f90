module integrate
   use iso_fortran_env, only: dp => real64
   use constants, only: c, charge_to_mass_ratio, num_particles, &
      integration_duration, integration_time_step, phi_0, tau_0
   use vector
   use beam, only: compute_laguerre_gauss_beam_electric_and_magnetic_field

   implicit none (type, external)

contains

   subroutine integrate_trajectories(initial_positions, initial_momenta, final_positions, final_momenta)
      real(kind=dp), dimension(4, num_particles), intent(in) :: &
         initial_positions(:, :), initial_momenta(:, :)

      real(kind=dp), dimension(4, num_particles), allocatable, intent(out) :: &
         final_positions(:, :), final_momenta(:, :)

      real(kind=dp), dimension(4, num_particles), allocatable :: positions(:, :), momenta(:, :)

      integer, parameter :: num_integration_steps = ceiling(integration_duration / integration_time_step)

      integer :: particle_index, step

      integer(kind=selected_int_kind(18)) :: rate, start_time, end_time
      real(kind=dp) :: duration

      allocate(positions(4, num_particles))
      allocate(momenta(4, num_particles))

      positions(:, :) = initial_positions
      momenta(:, :) = initial_momenta

      write(*, '("Starting to integrate trajectories")')

      call system_clock(count_rate=rate)

      call system_clock(start_time)

      !$omp parallel
      !$omp do
      do particle_index = 1, num_particles
         do step = 1, num_integration_steps
            call perform_integration_step(positions(1:4, particle_index), momenta(1:4, particle_index))
         end do
      end do
      !$omp end parallel

      call system_clock(end_time)

      duration = real(end_time - start_time, dp)/rate

      write(*, '("Done integrating trajectories")')
      write(*, '("Took ", F8.6, " seconds")') duration

      final_positions = positions
      final_momenta = momenta

   end subroutine integrate_trajectories

   pure subroutine perform_integration_step(position, momentum)
      real(kind=dp), intent(inout) :: position(1:4), momentum(1:4)

      type(vec4_t) :: previous_position, previous_momentum, &
         acceleration, new_position, new_momentum
      type(vec3_t) :: position_vector, E, B
      real(kind=dp) :: laboratory_time, coeff

      previous_position = to_vec4(position)
      previous_momentum = to_vec4(momentum)

      laboratory_time = previous_position%a

      position_vector = vec3_t(previous_position%x, previous_position%y, previous_position%z)

      call compute_laguerre_gauss_beam_electric_and_magnetic_field( &
         position_vector, laboratory_time, E, B)

      coeff = cutoff(laboratory_time - previous_position%z / c)
      E = coeff * E
      B = coeff * B

      acceleration = compute_acceleration(previous_momentum, E, B)

      new_momentum = previous_momentum + integration_time_step * acceleration
      new_position = previous_position + integration_time_step * new_momentum

      position(:) = vec4_to_array(new_position)
      momentum(:) = vec4_to_array(new_momentum)

   end subroutine

   pure function compute_acceleration(momentum, E, B) result(acc)
      type(vec4_t), intent(in) :: momentum
      type(vec3_t), intent(in) :: E, B

      type(vec4_t) :: p, acc

      p = momentum
      acc = charge_to_mass_ratio * vec4_t( &
         p%x * E%x / c + p%y * E%y/c + p%z * E%z/c, &
         p%a * E%x / c + p%y * B%z - p%z * B%y, &
         p%a * E%y / c - p%x * B%z + p%z * B%x, &
         p%a * E%z / c + p%x * B%y - p%y * B%x &
         )

   end function compute_acceleration

   pure function cutoff(phi)
      real(kind=dp), intent(in) :: phi
      real(kind=dp) :: t, cutoff

      t = (phi - phi_0) / tau_0

      cutoff = exp(-t**2)

   end function cutoff

end module integrate
