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

initial_positions = np.load("outputs/initial_positions.npy")

print("Plotting initial electron positions")

fig = plt.figure(figsize=(10, 6))
fig.suptitle("Initial electron positions")
plot_particle_positions(fig, initial_positions[:, 1:4])
fig.savefig(plots_directory / "initial_electron_positions.png")


particle_trajectory = np.load("outputs/particle_trajectory.npy")

print("Plotting sample electron trajectory")

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Electron trajectory")

ax.plot(particle_trajectory[:, 1] - np.mean(particle_trajectory[:, 1]), label="x")
ax.plot(particle_trajectory[:, 2] - np.mean(particle_trajectory[:, 2]), label="y")

ax.set_xlabel("Time $t$")
ax.set_ylabel("Displacement")

fig.legend()
ax.grid()
fig.tight_layout()

fig.savefig(plots_directory / "electron_trajectory.pdf")


detector_positions = np.load("outputs/detector_positions.npy")
electric_field = np.load("outputs/electric_field.npy")
magnetic_field = np.load("outputs/magnetic_field.npy")

print("Plotting final state detected electric field")
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Re-emitted radiation electric field")
# plt.plot(detector_positions[:, 0], np.linalg.vector_norm(electric_field, axis=-1))
# plt.plot(np.arange(len(detector_positions)), np.angle(electric_field[:, 0]))
plt.plot(detector_positions[:, 0], np.real(electric_field[:, 0]), label="Re $E_x$")
plt.plot(detector_positions[:, 0], np.imag(electric_field[:, 0]), label="Im $E_x$")
plt.plot(detector_positions[:, 0], np.real(electric_field[:, 1]), label="Re $E_y$")
plt.plot(detector_positions[:, 0], np.imag(electric_field[:, 1]), label="Im $E_y$")
plt.plot(detector_positions[:, 0], np.real(electric_field[:, 2]), label="Re $E_z$")
plt.plot(detector_positions[:, 0], np.imag(electric_field[:, 2]), label="Im $E_z$")
plt.legend()
plt.xlabel("Detector $x$ coordinate")
plt.ylabel("$E(x_0)$")
plt.grid()
fig.savefig(plots_directory / "electric_field.pdf")


print("Plotting final state detected magnetic field")
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Re-emitted radiation magnetic field")
# plt.plot(detector_positions[:, 0], np.linalg.vector_norm(electric_field, axis=-1))
plt.plot(detector_positions[:, 0], np.real(magnetic_field[:, 0]), label="Re $B_x$")
plt.plot(detector_positions[:, 0], np.imag(magnetic_field[:, 0]), label="Im $B_x$")
plt.plot(detector_positions[:, 0], np.real(magnetic_field[:, 1]), label="Re $B_y$")
plt.plot(detector_positions[:, 0], np.imag(magnetic_field[:, 1]), label="Im $B_y$")
plt.plot(detector_positions[:, 0], np.real(magnetic_field[:, 2]), label="Re $B_z$")
plt.plot(detector_positions[:, 0], np.imag(magnetic_field[:, 2]), label="Im $B_z$")
plt.legend()
plt.xlabel("Detector $x$ coordinate")
plt.ylabel("$B(x_0)$")
plt.grid()
fig.savefig(plots_directory / "magnetic_field.pdf")

final_angular_momenta = np.load("outputs/angular_momenta.npy")

print("Maximum angular momentum:", final_angular_momenta.max())

# Plot final angular momenta distribution
fig = plt.figure(dpi=200)

wavelength = lmbd
waist_radius = 75 * wavelength

plot_angular_momentum_distribution(
    fig, initial_positions[:, 1:4], waist_radius, final_angular_momenta
)

fig.savefig(plots_directory / "angular_momentum_distribution.png")
