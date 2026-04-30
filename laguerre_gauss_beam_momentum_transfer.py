import matplotlib.pyplot as plt
import numpy as np

from electrodynamics.beams import (
    PolarizationVector,
    compute_electric_and_magnetic_field_for_gaussian_beam,
    compute_electric_and_magnetic_field_for_laguerre_gauss_beam,
    compute_electric_and_magnetic_field_for_plane_wave,
)
from electrodynamics.constants import c, lmbd, omega_laser
from electrodynamics.initial_conditions import generate_initial_positions_on_disk
from electrodynamics.iterate import iterate_initial_conditions_hamiltonian_system
from electrodynamics.plotting import (
    plot_angular_momentum_distribution,
    plot_particle_positions,
)
from electrodynamics.tensor import compute_electromagnetic_field_tensor
from electrodynamics.typing import RealArray


def main() -> int:
    # Electron momentum transfer reproduction study:
    #   1. Generate a lot of electron (i.e. their initial position 4-vectors
    #     and their initial energy-momentum 4-vectors)
    #   2. Simulate their evolution for a fixed time span [0, T]
    #     (each electron evolves in its own proper time,
    #      with respect to the laboratory rest frame)
    #   3. Plot results:
    #     - Evolution of electron positions (coordinate projections of the 4-vectors)
    #     - Momentum distribution at the end (i.e. histogram of the velocities)

    rng = np.random.default_rng(seed=17)
    radius = 250 * lmbd
    num_particles = 8192
    initial_positions = generate_initial_positions_on_disk(rng, radius, num_particles)

    # Plot initial electron positions, for debugging purposes
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Initial electron positions")
    plot_particle_positions(fig, initial_positions)
    fig.savefig("plots/initial_electron_positions.pdf")

    initial_positions = np.hstack(
        (np.zeros((num_particles, 1), dtype=np.float64), initial_positions)
    )

    initial_velocities = np.zeros((num_particles, 4), dtype=np.float64)
    # Initial gamma = 1 (since particles are at rest)
    initial_velocities[:, 0] = 1

    # Electron charge and mass (in "natural" units)
    particle_charge = -1
    particle_mass = 1

    azimuthal_index = -2
    radial_index = 2

    minkowski_metric = np.diag(np.array((1, -1, -1, -1), dtype=np.float64))

    # beam_type = "plane_wave"
    # beam_type = "gaussian"
    beam_type = "laguerre_gauss"

    def cutoff(phi: RealArray, phi_0: float, tau_0: float) -> RealArray:
        return np.exp(-(((phi - phi_0) / tau_0) ** 2))

    tau_0 = 10 / omega_laser
    # tau_0 = 100 / omega_laser
    print("Tau 0 =", tau_0)
    phi_0 = 3 * tau_0
    integration_time = 6 * tau_0
    print("Integration time:", integration_time)

    phi = np.linspace(0, integration_time, 128, dtype=np.float64)

    plt.figure()
    plt.plot(phi, cutoff(phi, phi_0=phi_0, tau_0=tau_0))
    plt.savefig("plots/cutoff.pdf")
    plt.close()

    a_0 = 1e-2
    # E_0 = a_0 * m_e * c * omega / |q|
    amplitude = a_0 * particle_mass * c * omega_laser / abs(particle_charge)
    # polarization = PolarizationVector(1, 0)
    polarization = PolarizationVector(1 / np.sqrt(2), 1j / np.sqrt(2))
    wavelength = lmbd
    waist_radius = 75 * wavelength

    def acceleration(q: RealArray, p: RealArray) -> RealArray:
        laboratory_time = q[:, 0]
        position_vectors = q[:, 1:4]

        if beam_type == "plane_wave":
            E, B = compute_electric_and_magnetic_field_for_plane_wave(
                positions=position_vectors,
                time=laboratory_time,
            )
        elif beam_type == "gaussian":
            E, B = compute_electric_and_magnetic_field_for_gaussian_beam(
                polarization=polarization,
                positions=position_vectors,
                time=laboratory_time,
            )
        elif beam_type == "laguerre_gauss":
            E, B = compute_electric_and_magnetic_field_for_laguerre_gauss_beam(
                amplitude=amplitude,
                waist_radius=waist_radius,
                wavelength=wavelength,
                radial_index=radial_index,
                azimuthal_index=azimuthal_index,
                polarization=polarization,
                positions=position_vectors,
                time=laboratory_time,
            )
        else:
            raise NotImplementedError(f"unsupported beam type '{beam_type}'")

        cut = cutoff(laboratory_time - position_vectors[:, 2] / c, phi_0, tau_0)
        cut = cut.reshape(-1, 1)
        E *= cut
        B *= cut

        # print(E, B)

        field_tensor = compute_electromagnetic_field_tensor(E, B)

        v_lower_indices = p @ minkowski_metric

        # print("v_lower =", v_lower_indices)

        acceleration = (particle_charge / particle_mass) * (
            field_tensor @ v_lower_indices.reshape(len(q), 4, 1)
        ).squeeze()

        # print("a =", acceleration)

        # (F u, u) should be 0
        assert np.allclose(0, np.linalg.vecdot(acceleration, v_lower_indices.squeeze()))

        return acceleration

    times, positions, velocities = iterate_initial_conditions_hamiltonian_system(
        acceleration,
        initial_positions,
        initial_velocities,
        integration_time=integration_time,
        time_step=0.1,
    )

    final_positions = positions[-1, :, 1:4]
    final_velocities = velocities[-1, :, 1:4]

    angular_momentum_along_z_axis = azimuthal_index * (
        final_positions[:, 0] * final_velocities[:, 1]
        - final_positions[:, 1] * final_velocities[:, 0]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    for index in range(1):
        # axes[0].plot(times, positions[:, index, 0], label="$t$")
        axes[0].plot(times, positions[:, index, 1], label="$x$")
        axes[0].plot(times, positions[:, index, 2], label="$y$")
        axes[0].plot(times, positions[:, index, 3], label="$z$")

        axes[1].plot(times, velocities[:, index, 0], label="$c \\gamma$")
        axes[1].plot(times, velocities[:, index, 1], label="$v_x$")
        axes[1].plot(times, velocities[:, index, 2], label="$v_y$")
        axes[1].plot(times, velocities[:, index, 3], label="$v_z$")

    axes[0].grid()
    axes[0].legend()
    axes[0].set_xlabel("Proper time")

    axes[1].grid()
    axes[1].legend()
    axes[1].set_xlabel("Proper time")

    fig.savefig("plots/electron_trajectories.pdf")

    print("Maximum angular momentum:", angular_momentum_along_z_axis.max())

    plot_angular_momentum_distribution(
        initial_positions[:, 1:4], waist_radius, angular_momentum_along_z_axis
    )

    return 0


if __name__ == "__main__":
    exit(main())
