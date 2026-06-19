#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.angular_momentum import compute_angular_momentum_derivative
from electrodynamics.constants import lmbd, omega_laser
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

detector_positions = detector_positions.reshape(64, 64, 3)
electric_field = electric_field.reshape(64, 64, 3)
magnetic_field = magnetic_field.reshape(64, 64, 3)

print("Plotting final state detected electric field")
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Re-emitted radiation electric field")
plt.imshow(np.real(electric_field[:, :, 1]))
# plt.imshow(np.linalg.vector_norm(electric_field, axis=-1))
# plt.imshow(np.angle(electric_field[:, :, 0]))
# plt.legend()
plt.colorbar()
plt.xlabel("Detector $x$")
plt.ylabel("Detector $y$")
plt.grid()
fig.savefig(plots_directory / "electric_field.pdf")


print("Plotting final state detected magnetic field")
fig = plt.figure(figsize=(10, 6))
fig.suptitle("Re-emitted radiation magnetic field")
plt.imshow(np.real(magnetic_field[:, :, 1]))
# plt.imshow(np.linalg.vector_norm(magnetic_field, axis=-1))
# plt.imshow(np.angle(magnetic_field[:, :, 0]))
# plt.legend()
plt.colorbar()
plt.xlabel("Detector $x$")
plt.ylabel("Detector $y$")
plt.grid()
fig.savefig(plots_directory / "magnetic_field.pdf")

final_angular_momenta = np.load("outputs/angular_momenta.npy")

print("Plotting dL_z/dV")

fig = plt.figure(figsize=(10, 6))
fig.suptitle("$\\frac{dL_z}{dV}$")

dL_z = compute_angular_momentum_derivative(
    detector_positions, electric_field, omega_laser
)

plt.imshow(np.real(dL_z))
plt.colorbar()
plt.grid()
fig.savefig(plots_directory / "orbital_angular_momentum.pdf")

print("Maximum angular momentum:", final_angular_momenta.max())

# Plot final angular momenta distribution
fig = plt.figure(dpi=200)

wavelength = lmbd
waist_radius = 75 * wavelength

plot_angular_momentum_distribution(
    fig, initial_positions[:, 1:4], waist_radius, final_angular_momenta
)

fig.savefig(plots_directory / "angular_momentum_distribution.png")
