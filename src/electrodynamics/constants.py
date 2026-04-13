import numpy as np

### Physical constants (in atomic units)

# Speed of light
c = 137.036

# Angular frequency of laser radiation (we assume it's monochromatic)
omega_laser = 0.057

# Wavelength of laser radiation
lmbd = c * (2 * np.pi / omega_laser)
# print("Lambda:", lmbd)

# Number of particles (electrons) to simulate
num_particles = 30
