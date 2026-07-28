module integrate
   use iso_fortran_env, only: dp => real64
   use constants, only: c, charge_to_mass_ratio, &
      num_detector_points, num_particles, lambda, omega, pi, &
      num_integration_steps, integration_time_step, phi_0, tau_0
   use vector
   use beam, only: compute_laguerre_gauss_beam_electric_and_magnetic_field

   implicit none (type, external)

contains

   subroutine simulate_analytic_trajectories( &
      initial_positions, initial_momenta, detector_positions, &
      final_positions, final_momenta, electric_field &
      )
      real(kind=dp), intent(in) :: &
         initial_positions(1:4, 1:num_particles), initial_momenta(1:4,  1:num_particles)

      real(kind=dp), intent(in) :: &
         detector_positions(1:3, 1:num_detector_points)

      real(kind=dp), dimension(4, num_particles), allocatable, intent(out) :: &
         final_positions(:, :), final_momenta(:, :)

      complex(kind=dp), dimension(3, num_detector_points), allocatable, intent(out) :: &
         electric_field(:, :)

      real(kind=dp), dimension(4, num_particles), allocatable :: positions(:, :), momenta(:, :)

      integer :: particle_index, step

      integer(kind=selected_int_kind(18)) :: rate, start_time, end_time

      real(kind=dp) :: trajectory_amplitude, center_of_motion(1:3), phase_offset, &
         current_time, duration

      allocate(positions(4, num_particles))
      allocate(momenta(4, num_particles))

      positions(:, :) = initial_positions
      momenta(:, :) = initial_momenta

      allocate(electric_field(3, num_detector_points), source=complex(0.0_dp, 0.0_dp))

      write(*, '("Starting to analytically simulate particle trajectories")')

      call system_clock(count_rate=rate)

      call system_clock(start_time)

      !$omp parallel
      !$omp do private(center_of_motion, phase_offset, current_time) reduction(+:electric_field)
      do particle_index = 1, num_particles
         center_of_motion = initial_positions(2:4, particle_index)
         phase_offset = real(particle_index - 1, kind=dp) * (2 * pi) / real(num_particles, kind=dp)

         trajectory_amplitude = (0.1 / (2 * pi)) * lambda

         ! Initialize a variable to track the current time
         ! in the particle's local frame of reference
         current_time = 0

         do step = 1, num_integration_steps
            positions(2:4, particle_index) = [ &
               center_of_motion(1) + trajectory_amplitude * cos(omega * current_time - phase_offset), &
               center_of_motion(2) + trajectory_amplitude * sin(omega * current_time - phase_offset), &
               0.0_dp &
               ]
            momenta(2:4, particle_index) = [ &
               -omega * trajectory_amplitude * sin(omega * current_time - phase_offset), &
               omega * trajectory_amplitude * cos(omega * current_time - phase_offset), &
               0.0_dp &
               ]

            call compute_scattered_field( &
               detector_positions, center_of_motion, current_time, &
               positions(2:4, particle_index), momenta(2:4, particle_index), &
               electric_field)

            current_time = current_time + integration_time_step
         end do
      end do
      !$omp end do
      !$omp end parallel

      call system_clock(end_time)

      duration = real(end_time - start_time, dp)/rate

      write(*, '("Done integrating trajectories")')
      write(*, '("Took ", F10.6, " seconds")') duration

      final_positions = positions
      final_momenta = momenta

   end subroutine simulate_analytic_trajectories

   subroutine integrate_trajectories( &
      initial_positions, initial_momenta, detector_positions, &
      final_positions, final_momenta, electric_field &
      )
      real(kind=dp), intent(in) :: &
         initial_positions(1:4, 1:num_particles), initial_momenta(1:4,  1:num_particles)

      real(kind=dp), intent(in) :: &
         detector_positions(1:3, 1:num_detector_points)

      real(kind=dp), dimension(4, num_particles), allocatable, intent(out) :: &
         final_positions(:, :), final_momenta(:, :)

      complex(kind=dp), dimension(3, num_detector_points), allocatable, intent(out) :: &
         electric_field(:, :)

      real(kind=dp), dimension(4, num_particles), allocatable :: positions(:, :), momenta(:, :)

      integer :: particle_index, step

      integer(kind=selected_int_kind(18)) :: rate, start_time, end_time

      real(kind=dp) :: current_time, duration

      allocate(positions(4, num_particles))
      allocate(momenta(4, num_particles))

      positions(:, :) = initial_positions
      momenta(:, :) = initial_momenta

      allocate(electric_field(3, num_detector_points), source=complex(0.0_dp, 0.0_dp))

      write(*, '("Starting to integrate trajectories")')

      call system_clock(count_rate=rate)

      call system_clock(start_time)

      !$omp parallel
      !$omp do private(current_time) reduction(+:electric_field)
      do particle_index = 1, num_particles

         ! Initialize a variable to track the current time
         ! in the particle's local frame of reference
         current_time = 0

         do step = 1, num_integration_steps
            call perform_integration_step(positions(1:4, particle_index), momenta(1:4, particle_index))

            call compute_scattered_field( &
               detector_positions, initial_positions(2:4, particle_index), current_time, &
               positions(2:4, particle_index), momenta(2:4, particle_index), &
               electric_field)

            current_time = current_time + integration_time_step
         end do
      end do
      !$omp end do
      !$omp end parallel

      call system_clock(end_time)

      duration = real(end_time - start_time, dp)/rate

      write(*, '("Done integrating trajectories")')
      write(*, '("Took ", F10.6, " seconds")') duration

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

   pure subroutine compute_scattered_field(detector_positions, center_of_motion, time, position, velocity, electric_field)
      real(kind=dp), intent(in) :: detector_positions(1:3, 1:num_detector_points), &
         center_of_motion(1:3), time, &
         position(1:3), velocity(1:3)

      complex(kind=dp), intent(inout) :: electric_field(1:3, 1:num_detector_points)

      complex(kind=dp) :: oscillatory_kernel_term, first_order_term(1:3)

      integer :: detector_index
      real(kind=dp) :: detector_position(1:3), particle_position(1:3), &
         relative_detector_position(1:3), relative_particle_position(1:3), &
         displacement(1:3), displacement_norm, &
         relativistic_factor(1:3), view_direction(1:3)

      do detector_index = 1, num_detector_points
         detector_position = detector_positions(1:3, detector_index)
         particle_position = position

         ! x_0 = x - \symfrak{R}_0
         relative_detector_position = detector_position - center_of_motion

         ! r_0(t) = r(t) - \symfrak{R}_0
         relative_particle_position = particle_position - center_of_motion

         ! R(x_0, t) = x_0 - r_0(t)
         displacement = relative_detector_position - relative_particle_position

         ! |R(x_0, t)|
         displacement_norm = norm2(displacement)

         ! exp(i * omega * (t + |R(x_0, t)|/c))
         oscillatory_kernel_term = exp(cmplx(0.0_dp, 1.0_dp, kind=dp) * omega * (time + displacement_norm / c))

         ! beta = v / c
         relativistic_factor = velocity / c

         ! n = R(x_0, t) / |R(x_0, t)|
         view_direction = displacement / displacement_norm

         ! (i * omega) / c * (beta(t) - n(x_0, t) * dot(n, beta)) / |R(x_0, t)|
         first_order_term = (cmplx(0.0_dp, 1.0_dp, kind=dp) * omega / c) * &
            (relativistic_factor - view_direction * dot_product(view_direction, relativistic_factor)) / displacement_norm

         electric_field(:, detector_index) = electric_field(:, detector_index) + &
            integration_time_step * oscillatory_kernel_term * first_order_term
      end do

   end subroutine compute_scattered_field

end module integrate
