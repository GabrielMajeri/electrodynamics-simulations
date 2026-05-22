use std::f64::consts::PI;

use nalgebra::vector;
use num::Complex;

use crate::{
    constants::{
        AMPLITUDE, ANGULAR_VELOCITY, AZIMUTHAL_INDEX, POLARIZATION_X, POLARIZATION_Y, RADIAL_INDEX,
        SPEED_OF_LIGHT, WAIST_RADIUS, WAVELENGTH,
    },
    types::{Real, Vec3},
};

const TOLERANCE: Real = 1e-8;

/// Computes the electric and magnetic field vectors for a laser beam of Laguerre-Gauss shape,
/// at the given spatial position and time.
pub fn laguerre_gauss_beam_electric_and_magnetic_field(position: Vec3, time: Real) -> (Vec3, Vec3) {
    let rho = (position.x * position.x + position.y * position.y).sqrt();
    let phi = position.y.atan2(position.x);
    let x = position.x;
    let y = position.y;
    let z = position.z;

    let rayleigh_length = PI * WAIST_RADIUS.powi(2) / WAVELENGTH;

    // w(z)
    let width = WAIST_RADIUS * (1.0 + (z / rayleigh_length).powi(2)).sqrt();

    // rho / w(z)
    let rho_over_width = rho / width;
    let rho_over_width_squared = rho_over_width.powi(2);

    // k
    const WAVENUMBER: Real = 2.0 * PI / WAVELENGTH;

    // |l|
    const ABS_L: i8 = AZIMUTHAL_INDEX.abs();

    // R(z)
    let radius_of_curvature = if z.abs() < TOLERANCE {
        0.0
    } else {
        z * (1.0 + (rayleigh_length / z).powi(2))
    };

    // rho^2/(2 * R(z))
    let curvature = if radius_of_curvature < TOLERANCE {
        0.0
    } else {
        rho.powi(2) / (2.0 * radius_of_curvature)
    };

    // \psi(z)
    let gouy_phase = z.atan2(rayleigh_length);

    let magnitude = AMPLITUDE
        * (WAIST_RADIUS / width)
        * (2.0_f64.sqrt() * rho_over_width).powi(ABS_L.into())
        * laguerre_polynomial(RADIAL_INDEX, ABS_L.into(), 2.0 * rho_over_width_squared)
        * (-rho_over_width_squared).exp();

    let phase = Complex {
        re: 0.0,
        im: ANGULAR_VELOCITY * time - WAVENUMBER * z
            + WAVENUMBER * curvature
            + Real::from(AZIMUTHAL_INDEX) * phi
            - (2.0 * Real::from(RADIAL_INDEX) + Real::from(ABS_L) + 1.0) * gouy_phase,
    }
    .exp();

    let coeff = magnitude * phase;

    let e_x = coeff * POLARIZATION_X;
    let e_y = coeff * POLARIZATION_Y;
    let e_z = Complex { re: 0.0, im: 2.0 } / (WAVENUMBER * width.powi(2)) * (x * e_x + y * e_y);

    let e = vector![e_x.re, e_y.re, e_z.re];

    let b_x = -e_y / SPEED_OF_LIGHT;
    let b_y = e_x / SPEED_OF_LIGHT;
    let b_z = Complex::i() / (ANGULAR_VELOCITY * width.powi(2)) * (y * e_x - x * e_y);

    let b = vector![b_x.re, b_y.re, b_z.re];

    (e, b)
}

/// Laguerre gauss polynomial evaluation (fast for small n).
fn laguerre_polynomial(n: u8, alpha: Real, x: Real) -> Real {
    if n == 0 {
        return 1.0;
    }

    if n == 1 {
        return 1.0 + alpha - x;
    }

    if n == 2 {
        return 0.5 * (x.powi(2) - 2.0 * (alpha + 2.0) * x + (alpha + 1.0) * (alpha + 2.0));
    }

    let nr = n as Real;

    ((2.0 * nr - 1.0 + alpha - x) * laguerre_polynomial(n - 1, alpha, x)
        - (nr - 1.0 + alpha) * laguerre_polynomial(n - 2, alpha, x))
        / nr
}
