"""Numerical integration of the trajectory of
a single relativistic charged particle
(e.g. electron moving close to the speed of light).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.beams import (
    PolarizationVector,
    compute_electric_and_magnetic_field_for_laguerre_gauss_beam,
)
from electrodynamics.constants import c, lmbd, omega_laser
from electrodynamics.typing import RealArray

# Electron charge and mass (in "natural" units)
particle_charge = -1
particle_mass = 1

# Laser intensity factor
a_0 = 4

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

phi_0 = 3 * tau_0

integration_start_time = 0
integration_end_time = 6 * tau_0
integration_duration = integration_end_time - integration_start_time

charge_to_mass_ratio = particle_charge / particle_mass


def main() -> None:
    time_step = tau_0 / 10_000
    num_integration_steps = int(integration_duration / time_step) + 1
    print(f"Number of steps for integration: {num_integration_steps}")

    initial_position = np.array([0.0, waist_radius / 4, 0.0, 0.0], dtype=np.float64)
    initial_momentum = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    positions = np.empty((num_integration_steps, 4))
    momenta = np.empty((num_integration_steps, 4))

    positions[0] = initial_position[:]
    momenta[0] = initial_momentum[:]

    print("Integrating relativistic particle trajectory")
    dot_product = 0

    for step in range(1, num_integration_steps):
        previous_position = positions[step - 1]
        previous_momentum = momenta[step - 1]

        laboratory_time = previous_position[0]
        position_vector = previous_position[1:4]

        # Runge-Kutta method of order 4 (RK4)
        k_1 = compute_acceleration(
            laboratory_time,
            position_vector,
            previous_momentum,
        )
        k_2 = compute_acceleration(
            laboratory_time + time_step / 2,
            position_vector,
            previous_momentum + time_step / 2 * k_1,
        )
        k_3 = compute_acceleration(
            laboratory_time + time_step / 2,
            position_vector,
            previous_momentum + time_step / 2 * k_2,
        )
        k_4 = compute_acceleration(
            laboratory_time + time_step,
            position_vector,
            previous_momentum + time_step * k_3,
        )

        acceleration = (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

        new_momentum = previous_momentum + time_step * acceleration
        new_position = previous_position + time_step * new_momentum

        # \\gamma should always be >= 1
        assert new_momentum[0] >= 1

        # (F u, u) should be 0
        dot_product = max(dot_product, np.dot(acceleration, new_momentum))
        # BUG: still goes up to ~0.01 during numerical integration
        # print(np.dot(acceleration, new_momentum))
        # assert np.allclose(0, np.dot(acceleration, new_momentum), atol=1e-2, rtol=1e-2)

        positions[step] = new_position
        momenta[step] = new_momentum

    print("Dot product max:", dot_product)

    plots_directory = Path("plots")
    plots_directory.mkdir(parents=True, exist_ok=True)

    print("Plotting particle trajectory")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Electron trajectory")

    t_ax = ax.twinx()

    t_ax.plot(positions[:, 0], label="t", color="yellow")
    ax.plot(positions[:, 1] - np.mean(positions[:, 1]), label="x")
    ax.plot(positions[:, 2] - np.mean(positions[:, 2]), label="y")
    ax.plot(positions[:, 3] - np.mean(positions[:, 3]), label="z")

    ax.set_xlabel("Proper time $t$")
    ax.set_ylabel("Displacement")
    t_ax.set_ylabel("Laboratory time $\\tau$")

    fig.legend()
    ax.grid()
    fig.tight_layout()

    fig.savefig(plots_directory / "particle_trajectory.pdf")

    print("Plotting particle momenta")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Electron momenta")

    t_ax = ax.twinx()

    t_ax.plot(momenta[:, 0], label="$\\gamma$", color="red")
    ax.plot(momenta[:, 1], label="$p_x$")
    ax.plot(momenta[:, 2], label="$p_y$")
    ax.plot(momenta[:, 3], label="$p_z$")

    ax.set_xlabel("Proper time $t$")
    ax.set_ylabel("Displacement")
    t_ax.set_ylabel("Relativistic factor")

    fig.legend()
    ax.grid()
    fig.tight_layout()

    fig.savefig(plots_directory / "particle_momenta.pdf")


def cutoff(phi: float, phi_0: float, tau_0: float) -> float:
    parameter = (phi - phi_0) / tau_0
    argument = -(parameter**2)
    return np.exp(argument)


def compute_acceleration(
    time: float, position: RealArray, momentum: RealArray
) -> RealArray:
    E = np.empty((3,), dtype=np.float64)
    B = np.empty((3,), dtype=np.float64)

    compute_electric_and_magnetic_field_for_laguerre_gauss_beam(
        amplitude=amplitude,
        waist_radius=waist_radius,
        wavelength=wavelength,
        radial_index=radial_index,
        azimuthal_index=azimuthal_index,
        polarization=polarization_arr,
        position=position,
        time=time,
        E=E,
        B=B,
    )

    cut = cutoff(time - position[2] / c, phi_0, tau_0)
    E *= cut
    B *= cut

    acceleration = np.zeros((4,), dtype=np.float64)

    v = momentum
    acceleration[0] = charge_to_mass_ratio * (
        v[1] * E[0] / c + v[2] * E[1] / c + v[3] * E[2] / c
    )
    acceleration[1] = charge_to_mass_ratio * (
        v[0] * E[0] / c + v[2] * B[2] - v[3] * B[1]
    )
    acceleration[2] = charge_to_mass_ratio * (
        v[0] * E[1] / c - v[1] * B[2] + v[3] * B[0]
    )
    acceleration[3] = charge_to_mass_ratio * (
        v[0] * E[2] / c + v[1] * B[1] - v[2] * B[0]
    )

    return acceleration


if __name__ == "__main__":
    main()
