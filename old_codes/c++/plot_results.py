#!/usr/bin/env python3

from argparse import ArgumentParser, BooleanOptionalAction
from math import pi
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from electrodynamics.angular_momentum import compute_angular_momentum_derivative
from electrodynamics.constants import SPEED_OF_LIGHT
from electrodynamics.plotting import (
    plot_angular_momentum_distribution,
    plot_particle_positions,
)

c = SPEED_OF_LIGHT
omega_laser = 0.057
lmbd = (2 * pi * c) / omega_laser

parser = ArgumentParser("plot_results.py")

parser.add_argument(
    "--title",
    action=BooleanOptionalAction,
    help="Include the title in each plot.",
)

args = parser.parse_args()

with_title = args.title

plots_directory = Path("plots")
plots_directory.mkdir(parents=True, exist_ok=True)

initial_positions = np.load("outputs/initial_positions.npy")

### Initial particle positions plot ###

print("Plotting initial electron positions")

fig = plt.figure(figsize=(10, 6))
if with_title:
    fig.suptitle("Initial electron positions")
plot_particle_positions(fig, initial_positions[:, 1:4])
fig.savefig(plots_directory / "initial_electron_positions.png")


### Sample electron trajectory plot ###

print("Plotting sample electron trajectory")

particle_trajectory = np.load("outputs/particle_trajectory.npy")

fig, ax = plt.subplots(figsize=(10, 6))
if with_title:
    fig.suptitle("Electron trajectory")

twin_ax = ax.twinx()

twin_ax.plot(particle_trajectory[:, 0], label="$\\tau$", color="cyan")
ax.plot(particle_trajectory[:, 1] - np.mean(particle_trajectory[:, 1]), label="$x$")
ax.plot(particle_trajectory[:, 2] - np.mean(particle_trajectory[:, 2]), label="$y$")
ax.plot(particle_trajectory[:, 3] - np.mean(particle_trajectory[:, 3]), label="$z$")

ax.set_xlabel("Proper time $t$")
ax.set_ylabel("Displacement")

fig.legend()
ax.grid()
fig.tight_layout()

fig.savefig(plots_directory / "electron_trajectory.pdf")


### Sample electron momenta plot ###

print("Plotting sample electron momenta")

particle_momenta = np.load("outputs/particle_momenta.npy")

fig, ax = plt.subplots(figsize=(10, 6))
if with_title:
    fig.suptitle("Electron momenta")

t_ax = ax.twinx()

t_ax.plot(particle_momenta[:, 0] / c, label="$\\gamma$", color="red")
ax.plot(particle_momenta[:, 1], label="$p_x$")
ax.plot(particle_momenta[:, 2], label="$p_y$")
ax.plot(particle_momenta[:, 3], label="$p_z$")

ax.set_xlabel("Proper time $t$")
ax.set_ylabel("Momentum")
t_ax.set_ylabel("Relativistic factor")

fig.legend()
ax.grid()
fig.tight_layout()

fig.savefig(plots_directory / "electron_momenta.pdf")

detector_positions = np.load("outputs/detector_positions.npy")
electric_field = np.load("outputs/electric_field.npy")
magnetic_field = np.load("outputs/magnetic_field.npy")

detector_nx = 64
detector_ny = 64

detector_positions = detector_positions.reshape(detector_nx, detector_ny, 3)
electric_field = electric_field.reshape(detector_nx, detector_ny, 3)
magnetic_field = magnetic_field.reshape(detector_nx, detector_ny, 3)


### Final state detector electric field plot ###

print("Plotting final state detected electric field")

for index, coordinate in enumerate(("x", "y", "z")):
    fig = plt.figure(figsize=(7, 6))
    if with_title:
        fig.suptitle("Scattered electric field on detector")
    plt.title(f"$E_{coordinate}$")
    plt.imshow(np.real(electric_field[:, :, index]), cmap="bwr")
    # plt.imshow(np.linalg.vector_norm(electric_field, axis=-1))
    # plt.imshow(np.angle(electric_field[:, :, 0]))
    # plt.legend()
    plt.colorbar()
    plt.xlabel("Detector $x$")
    plt.ylabel("Detector $y$")
    plt.grid()
    fig.savefig(plots_directory / f"electric_field_{coordinate}.pdf")


### Final state detector magnetic field plot ###

print("Plotting final state detected magnetic field")

for index, coordinate in enumerate(("x", "y", "z")):
    fig = plt.figure(figsize=(7, 6))
    if with_title:
        fig.suptitle("Scattered magnetic field on detector")
    plt.title(f"$B_{coordinate}$")
    plt.imshow(np.real(magnetic_field[:, :, index]), cmap="bwr")
    # plt.imshow(np.linalg.vector_norm(magnetic_field, axis=-1))
    # plt.imshow(np.angle(magnetic_field[:, :, 0]))
    # plt.legend()
    plt.colorbar()
    plt.xlabel("Detector $x$")
    plt.ylabel("Detector $y$")
    plt.grid()
    fig.savefig(plots_directory / f"magnetic_field_{coordinate}.pdf")


### Detector angular momenta transfer plot ###

final_angular_momenta = np.load("outputs/angular_momenta.npy")

print("Plotting dL_z/dV")

fig = plt.figure(figsize=(10, 6))
if with_title:
    fig.suptitle("$\\Re\\left(\\frac{dL_z}{dz}\\right)$")

dL_z = compute_angular_momentum_derivative(
    detector_positions, electric_field, omega_laser
)

plt.imshow(np.real(dL_z), cmap="bwr")
plt.colorbar()
plt.xlabel("Detector $x$")
plt.ylabel("Detector $y$")
plt.grid()
fig.savefig(plots_directory / "orbital_angular_momentum.pdf")


### Final state particle angular momentum distribution ###

print("Maximum angular momentum:", final_angular_momenta.max())

# Plot final angular momenta distribution
fig = plt.figure(dpi=200)

wavelength = lmbd
waist_radius = 75 * wavelength

plot_angular_momentum_distribution(
    fig,
    initial_positions[:, 1:4],
    waist_radius,
    final_angular_momenta,
    with_title=with_title,
)

fig.savefig(plots_directory / "angular_momentum_distribution.png")
