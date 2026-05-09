use std::f64::consts::{PI, SQRT_2};

use num::Complex;

use crate::types::Real;

/// Seed for the random number generator.
pub const RANDOM_SEED: u64 = 1234;

/// Number of particles (electrons) to be simulated.
pub const NUM_PARTICLES: usize = 4 * 1024;

/// Speed of light in vacuum (in atomic units) -- basically the same as
/// the fine structure constant in this case.
pub const SPEED_OF_LIGHT: Real = 137.036;

/// Charge of the electron (in atomic units)
pub const PARTICLE_CHARGE: Real = -1.0;

/// Mass of the electron (in atomic units)
pub const PARTICLE_MASS: Real = 1.0;

/// Angular velocity of laser beam
pub const ANGULAR_VELOCITY: Real = 0.057;

/// Wavelength of monochromatic laser light (in atomic units)
pub const WAVELENGTH: Real = 2.0 * PI * SPEED_OF_LIGHT / ANGULAR_VELOCITY;

const A_0: Real = 1e-2;
pub const AMPLITUDE: Real =
    A_0 * PARTICLE_MASS * SPEED_OF_LIGHT * ANGULAR_VELOCITY / PARTICLE_CHARGE.abs();

/// Radius at beam waist (z = 0)
pub const WAIST_RADIUS: Real = 75.0 * WAVELENGTH;

/// Radial index (parameter "p" in formulae)
pub const RADIAL_INDEX: u8 = 2;
/// Azimuthal index (parameter "l" in formulae)
pub const AZIMUTHAL_INDEX: i8 = 2;

pub const DISK_RADIUS: Real = (1.75 + (RADIAL_INDEX as Real)) * WAIST_RADIUS;

/// Time duration of laser pulse (Gaussian envelope)
pub const TAU_0: Real = 10.0 / ANGULAR_VELOCITY;

/// Center of the laser's temporal Gaussian envelope
pub const PHI_0: Real = 3.0 * TAU_0;

/// Initial time for the integration
pub const INTEGRATION_START_TIME: Real = 0.0;
/// Final time for the integration
pub const INTEGRATION_END_TIME: Real = 6.0 * TAU_0;
/// Temporal discretization step
pub const INTEGRATION_TIME_STEP: Real = 0.1;

/// Total duration of the integration
pub const INTEGRATION_DURATION: Real = INTEGRATION_END_TIME - INTEGRATION_START_TIME;

/// Number of integration steps we will take
pub const NUM_INTEGRATION_STEPS: usize = (INTEGRATION_DURATION / INTEGRATION_TIME_STEP).ceil() as _;

/// Ratio between particle's charge (coupling strength to EM field) and its mass.
pub const CHARGE_TO_MASS_RATIO: Real = PARTICLE_CHARGE / PARTICLE_MASS;

/// Polarization component in the x direction.
pub const POLARIZATION_X: Complex<Real> = Complex {
    re: 1.0 / SQRT_2,
    im: 0.0,
};

/// Polarization component in the y direction.
pub const POLARIZATION_Y: Complex<Real> = Complex {
    re: 0.0,
    im: 1.0 / SQRT_2,
};
