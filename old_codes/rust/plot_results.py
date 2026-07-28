#!/usr/bin/env python3

from math import pi
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.constants import SPEED_OF_LIGHT
from electrodynamics.plotting import (
    plot_angular_momentum_distribution,
    plot_particle_positions,
)

c = SPEED_OF_LIGHT
omega_laser = 0.057
lmbd = (2 * pi * c) / omega_laser

plots_directory = Path("plots")
plots_directory.mkdir(parents=True, exist_ok=True)

initial_positions = np.load("initial_positions.npy")

# Plot initial electron positions
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Initial electron positions")
plot_particle_positions(fig, initial_positions[:, 1:4])
fig.savefig(plots_directory / "initial_electron_positions.png")


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
