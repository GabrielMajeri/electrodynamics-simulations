import matplotlib.pyplot as plt
import numpy as np

from electrodynamics.beams import (
    compute_electric_and_magnetic_field_for_laguerre_gauss_beam,
)
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
    radius = 5
    num_particles = 32
    initial_positions = generate_initial_positions_on_disk(rng, radius, num_particles)

    # Plot initial electron positions, for debugging purposes
    fig = plt.figure()
    fig.suptitle("Initial electron positions")
    plot_particle_positions(fig, initial_positions)
    fig.savefig("plots/initial_electron_positions.pdf")

    initial_positions = np.hstack(
        (np.zeros((num_particles, 1), dtype=np.float64), initial_positions)
    )

    initial_velocities = np.zeros((num_particles, 4), dtype=np.float64)

    # TODO: pick some realistic values
    particle_charge = 1
    particle_mass = 1

    azimuthal_index = -2
    radial_index = 2

    minkowski_metric = np.diag(np.array((1, -1, -1, -1), dtype=np.float64))

    def acceleration(q: RealArray, p: RealArray) -> RealArray:
        laboratory_time = q[:, 0]
        position_vectors = q[:, 1:4]

        # E, B = compute_electric_and_magnetic_field_for_plane_wave(
        #     position_vectors, laboratory_time
        # )

        # E, B = compute_electric_and_magnetic_field_for_gaussian_beam(
        #     position_vectors, laboratory_time
        # )

        E, B = compute_electric_and_magnetic_field_for_laguerre_gauss_beam(
            position_vectors, laboratory_time, l=azimuthal_index, p=radial_index
        )

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
        acceleration, initial_positions, initial_velocities, integration_time=10
    )

    final_positions = positions[-1, :, 1:4]
    final_velocities = velocities[-1, :, 1:4]

    angular_momentum_along_z_axis = azimuthal_index * (
        final_positions[:, 0] * final_velocities[:, 1]
        - final_positions[:, 1] * final_velocities[:, 0]
    )

    plot_angular_momentum_distribution(
        initial_positions[:, 1:4], angular_momentum_along_z_axis
    )

    return 0


if __name__ == "__main__":
    exit(main())
