use std::f64::consts::PI;

use crate::{
    constants::{DISK_RADIUS, NUM_PARTICLES, RANDOM_SEED},
    types::Vec4,
};
use nalgebra::vector;
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};

pub struct InitialConditions {
    pub positions: Vec<Vec4>,
    pub momenta: Vec<Vec4>,
}

/// Generate initial particle positions and momenta.
pub fn generate_initial_conditions() -> InitialConditions {
    let mut rng = SmallRng::seed_from_u64(RANDOM_SEED);

    let mut positions = Vec::with_capacity(NUM_PARTICLES);
    let mut momenta = Vec::with_capacity(NUM_PARTICLES);

    let radius_squared_distr = Uniform::new(0.0, DISK_RADIUS.powi(2)).unwrap();
    let angle_distr = Uniform::new(0.0, 2.0 * PI).unwrap();

    for _index in 0..NUM_PARTICLES {
        // Generate particles uniformly distributed in a disk of fixed radius,
        // centered at the origin

        let r = radius_squared_distr.sample(&mut rng).sqrt();
        let theta = angle_distr.sample(&mut rng);

        let x = r * theta.cos();
        let y = r * theta.sin();

        positions.push(vector![0.0, x, y, 0.0]);

        // All particles are initially at rest
        // (hence, the Lorentz factor \gamma is 1)
        momenta.push(vector![1.0, 0.0, 0.0, 0.0]);
    }

    InitialConditions { positions, momenta }
}
