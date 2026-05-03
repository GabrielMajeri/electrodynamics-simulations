from time import perf_counter

import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np

from electrodynamics.beams import (
    PolarizationVector,
    compute_electric_and_magnetic_field_for_gaussian_beam,
    compute_electric_and_magnetic_field_for_laguerre_gauss_beam,
    compute_electric_and_magnetic_field_for_plane_wave,
)
from electrodynamics.constants import c, lmbd, omega_laser
from electrodynamics.initial_conditions import generate_initial_positions_on_disk
from electrodynamics.plotting import (
    plot_angular_momentum_distribution,
    plot_particle_positions,
)
from electrodynamics.tensor import compute_electromagnetic_field_tensor
from electrodynamics.typing import RealArray

num_particles = 2 * 1024

# Electron charge and mass (in "natural" units)
particle_charge = -1
particle_mass = 1

# beam_type = "plane_wave"
# beam_type = "gaussian"
beam_type = "laguerre_gauss"

a_0 = 1e-2

# E_0 = a_0 * m_e * c * omega / |q|
amplitude = a_0 * particle_mass * c * omega_laser / abs(particle_charge)
wavelength = lmbd
waist_radius = 75 * wavelength

# polarization = PolarizationVector(1, 0)
polarization = PolarizationVector(1 / np.sqrt(2), 1j / np.sqrt(2))
polarization_arr = polarization.to_numpy_array()

azimuthal_index = -2
radial_index = 2

tau_0 = 10 / omega_laser
# tau_0 = 100 / omega_laser

integration_start_time = 0
integration_end_time = 6 * tau_0
integration_duration = integration_end_time - integration_start_time

time_step = tau_0 / 2048
num_integration_steps = int(integration_duration / time_step) + 1

phi_0 = 3 * tau_0

minkowski_metric = np.diag(np.array((1, -1, -1, -1), dtype=np.float64))


def main() -> int:
    rng = np.random.default_rng(seed=17)
    initial_positions, initial_momenta = generate_initial_conditions(rng)

    print(initial_positions.shape)

    # Plot initial electron positions, for debugging purposes
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Initial electron positions")
    plot_particle_positions(fig, initial_positions[:, 1:4])
    fig.savefig("plots/initial_electron_positions.pdf")

    print("Tau_0 =", tau_0)

    print("Integration duration:", integration_duration)
    print(
        f"Integration from t_initial = {integration_start_time} to t_final = {integration_end_time}"
    )
    print(f"Time step: {time_step}")
    print(f"Integration steps: {num_integration_steps}")

    phi = np.linspace(
        integration_start_time, integration_end_time, 128, dtype=np.float64
    )

    plt.figure()
    plt.plot(phi, cutoff(phi, phi_0=phi_0, tau_0=tau_0))
    plt.savefig("plots/cutoff.pdf")
    plt.close()

    print("Integrating electron motion in EM field due to laser pulse...")
    start_time = perf_counter()

    final_positions, final_momenta = integrate_trajectories(
        initial_positions, initial_momenta
    )

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Simulating trajectories took {duration} seconds")

    final_positions = final_positions[:, 1:4]
    final_velocities = final_momenta[:, 1:4]

    angular_momentum_along_z_axis = particle_mass * (
        final_positions[:, 0] * final_velocities[:, 1]
        - final_positions[:, 1] * final_velocities[:, 0]
    )

    print("Maximum angular momentum:", angular_momentum_along_z_axis.max())

    fig = plt.figure(dpi=200)

    plot_angular_momentum_distribution(
        fig, initial_positions[:, 1:4], waist_radius, angular_momentum_along_z_axis
    )

    fig.savefig("plots/angular_momentum_distribution.png")

    return 0


@njit(cache=True)
def generate_initial_conditions(
    rng: np.random.Generator,
) -> tuple[RealArray, RealArray]:
    radius = 250 * lmbd

    initial_positions = generate_initial_positions_on_disk(rng, radius, num_particles)

    initial_positions = np.hstack(
        (np.zeros((num_particles, 1), dtype=np.float64), initial_positions)
    )

    initial_momenta = np.zeros((num_particles, 4), dtype=np.float64)
    # Initial gamma = 1 (since particles are at rest)
    initial_momenta[:, 0] = 1

    return initial_positions, initial_momenta


@njit(cache=True, parallel=True)
def integrate_trajectories(
    initial_positions: RealArray, initial_momenta: RealArray
) -> tuple[RealArray, RealArray]:
    positions = initial_positions
    momenta = initial_momenta

    for particle_index in prange(initial_positions.shape[0]):
        for step in range(num_integration_steps):
            previous_position = positions[particle_index]
            previous_momentum = momenta[particle_index]

            laboratory_time = previous_position[0]
            position_vector = previous_position[1:4]

            if beam_type == "plane_wave":
                E, B = compute_electric_and_magnetic_field_for_plane_wave(
                    position=position_vector,
                    time=laboratory_time,
                )
            elif beam_type == "gaussian":
                E, B = compute_electric_and_magnetic_field_for_gaussian_beam(
                    polarization=polarization_arr,
                    position=position_vector,
                    time=laboratory_time,
                )
            elif beam_type == "laguerre_gauss":
                E, B = compute_electric_and_magnetic_field_for_laguerre_gauss_beam(
                    amplitude=amplitude,
                    waist_radius=waist_radius,
                    wavelength=wavelength,
                    radial_index=radial_index,
                    azimuthal_index=azimuthal_index,
                    polarization=polarization_arr,
                    position=position_vector,
                    time=laboratory_time,
                )
            else:
                raise NotImplementedError(f"unsupported beam type '{beam_type}'")

            cut = cutoff(laboratory_time - position_vector[2] / c, phi_0, tau_0)
            E *= cut
            B *= cut

            field_tensor = compute_electromagnetic_field_tensor(E, B)

            v_lower_indices = minkowski_metric @ previous_momentum

            acceleration = (particle_charge / particle_mass) * (
                field_tensor @ v_lower_indices
            )

            # TODO: breaks numba parallelization
            # (F u, u) should be 0
            # assert np.allclose(
            #     0,
            #     np.dot(acceleration, v_lower_indices),
            # )

            new_momentum = previous_momentum + time_step * acceleration
            new_position = previous_position + time_step * new_momentum

            positions[particle_index] = new_position
            momenta[particle_index] = new_momentum

    return positions, momenta


@njit(cache=True)
def cutoff(phi: float | RealArray, phi_0: float, tau_0: float) -> float | RealArray:
    parameter = (phi - phi_0) / tau_0
    argument = -(parameter**2)
    return np.exp(argument)


if __name__ == "__main__":
    exit(main())
