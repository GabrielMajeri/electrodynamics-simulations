from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.constants import lmbd
from electrodynamics.plotting import (
    plot_angular_momentum_distribution,
    plot_particle_positions,
)


initial_positions = np.load("initial_positions.npy")
final_angular_momenta = np.load("angular_momenta.npy")

plots_directory = Path("plots")
plots_directory.mkdir(parents=True, exist_ok=True)

# Plot initial electron positions
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Initial electron positions")
plot_particle_positions(fig, initial_positions[:, 1:4])
fig.savefig(plots_directory / "initial_electron_positions.png")


print("Maximum angular momentum:", final_angular_momenta.max())

# # Plot final angular momenta distribution
fig = plt.figure(dpi=200)

wavelength = lmbd
waist_radius = 75 * wavelength

plot_angular_momentum_distribution(
    fig, initial_positions[:, 1:4], waist_radius, final_angular_momenta
)

fig.savefig(plots_directory / "angular_momentum_distribution.png")
