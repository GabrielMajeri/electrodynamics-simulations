import numpy as np
import matplotlib.pyplot as plt

### Physical constants (in atomic units)
# Speed of light
c = 137.036

# Angular frequency of laser radiation (we assume it's monochromatic)
omega_laser = 0.057

# Wavelength of laser radiation
lmbd = c * (2 * np.pi / omega_laser)
# print("Lambda:", lmbd)

# Number of particles (electrons) to simulate
num_particles = 10

### Code to simulate numerically the particle's trajectories
### (they move around in a small circular trajectory around their center,
### and their centers are radially distributed around the origin)

large_circle_radius = 50 * lmbd

# How many timestamps to use (will affect trajectory sampling frequency)
num_timestamps = 1024
integration_time = 10 * (2 * np.pi) / omega_laser
timestamps = np.linspace(0, integration_time, num_timestamps)

particle_indices = np.arange(num_particles, dtype=np.int32)

# Determine the particles' center-of-motion
centers = np.array(
    (
        large_circle_radius * np.cos(2 * np.pi * particle_indices / num_particles),
        large_circle_radius * np.sin(2 * np.pi * particle_indices / num_particles),
        np.zeros(num_particles),
    )
).T

trajectories = np.empty((num_particles, num_timestamps, 3))

# We apply a decay to the trajectories to ensure the integral decays near the boundaries
# TODO: use a better envelope
trajectory_radius = (
    5
    * lmbd
    * np.exp(-((timestamps - integration_time / 2) ** 2) / (integration_time**2 / 32))
)

trajectories[:, :, 0] = (
    centers[:, 0, np.newaxis]
    + (trajectory_radius * np.cos(omega_laser * timestamps))[np.newaxis, :]
)

trajectories[:, :, 1] = (
    centers[:, 1, np.newaxis]
    + (trajectory_radius * np.sin(omega_laser * timestamps))[np.newaxis, :]
)

trajectories[:, :, 2] = 0

# Plot the particle's trajectories

if False:
    for particle in range(num_particles):
        plt.plot(
            trajectories[particle, :, 0],
            trajectories[particle, :, 1],
            label=f"Electron #{particle + 1}",
        )

    plt.gca().set_aspect(1)
    plt.grid()
    plt.legend()
    plt.show()

# Plot the current setup (particles and detector)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

for particle in range(num_particles):
    x, y, z = (
        trajectories[particle, :, 0],
        trajectories[particle, :, 1],
        trajectories[particle, :, 2],
    )
    ax.plot(
        x,
        y,
        z,
    )


detector_position = np.array([0, 0, 1000 * lmbd])

ax.scatter(*detector_position, s=10)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
plt.savefig("setup.pdf")
# plt.show()
plt.close()

# Current offset of particle from "center" of its motion
particle_displacements = trajectories - centers[:, np.newaxis, :]
r_0s = particle_displacements

plt.title("$r_0(t)$ for particle #1")
plt.plot(timestamps, particle_displacements[0, :, 0], label="x")
plt.plot(timestamps, particle_displacements[0, :, 1], label="y")
plt.legend()
plt.grid()
plt.savefig("r_0.pdf")
plt.close()

# Compute detector displacement (in each particle frame of reference)
x_0s = detector_position - centers
x_0s_norms = np.linalg.vector_norm(x_0s, axis=-1)

# Inverse of the distance between the particle's center of motion and the detector
# We expand in a Taylor series in terms of this value (should be very small)
print("1/|x_0|:", 1 / x_0s_norms[0])

n_0s = x_0s / x_0s_norms[:, np.newaxis]

# print(n_0s[0])

# TODO: look at a grid of frequencies, around the frequency of the laser
frequency = omega_laser * 1.00
n_0_dot_r_0 = np.vecdot(n_0s[0], r_0s[0])
exponent = frequency * timestamps - frequency / c * n_0_dot_r_0

# Plot (real-valued) exponent of oscillatory kernel
plt.title("Exponent ($g(t)$)")
plt.plot(timestamps, exponent)
plt.grid()
plt.savefig("g.pdf")
plt.close()

oscillatory_kernel = np.exp(1j * exponent)

plt.title("Oscillatory kernel")
plt.plot(timestamps, np.angle(oscillatory_kernel), marker=".", linewidth=0)
plt.xlabel("$t$")
plt.ylabel("")
plt.grid()
plt.savefig("exp_i_g.pdf")
plt.close()

# The integrand is now the oscillatory kernel times the derivative of the position term
integrand = oscillatory_kernel * np.gradient(n_0_dot_r_0)

plt.title("Integrand")
plt.plot(timestamps, np.abs(integrand), marker=".", linewidth=0)
plt.xlabel("$t$")
plt.ylabel("")
plt.grid()
plt.savefig("integrand.pdf")
plt.close()


# TODO: use an integration method specialized for highly-oscillatory integrals

dt = timestamps[1] - timestamps[0]
# integral = (1 / x_0s_norms[0]) * dt * np.sum(np.exp(1j * exponent))
integral = dt * np.sum(integrand)
print("Integral value:", integral)
print("Integral absolute value:", np.abs(integral))
