#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.constants import lmbd
from electrodynamics.plotting import (
    plot_angular_momentum_distribution,
    plot_particle_positions,
)


plots_directory = Path("plots")
plots_directory.mkdir(parents=True, exist_ok=True)

initial_positions = np.load("initial_positions.npy")

print("Plotting initial electron positions")

fig = plt.figure(figsize=(10, 6))
fig.suptitle("Initial electron positions")
plot_particle_positions(fig, initial_positions[:, 1:4])
fig.savefig(plots_directory / "initial_electron_positions.png")


detector_positions = np.load("detector_positions.npy")
electric_field = np.load("electric_field.npy")

n = detector_positions.shape[0]
detector_grid_size_x = int(np.sqrt(n))
detector_grid_size_y = int(np.sqrt(n))
detector_positions = detector_positions.reshape(
    detector_grid_size_y, detector_grid_size_x, 3
)
electric_field = electric_field.reshape(detector_grid_size_y, detector_grid_size_x, 3)

print("Plotting final state detected electric field")
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Re-emitted radiation electric field")
plt.imshow(np.linalg.vector_norm(electric_field, axis=-1))
plt.xlabel("Detector $x$ coordinate")
plt.ylabel("$E(x_0)$")
plt.grid()
fig.savefig(plots_directory / "electric_field.png")


final_angular_momenta = np.load("angular_momenta.npy")

print("Maximum angular momentum:", final_angular_momenta.max())

# Plot final angular momenta distribution
fig = plt.figure(dpi=200)

wavelength = lmbd
waist_radius = 75 * wavelength

plot_angular_momentum_distribution(
    fig, initial_positions[:, 1:4], waist_radius, final_angular_momenta
)

fig.savefig(plots_directory / "angular_momentum_distribution.png")
