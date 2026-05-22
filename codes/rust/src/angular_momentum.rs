use crate::types::{Real, Vec4};

/// Compute the angular momenta (in the z direction) of a bunch of particles.
pub fn compute_angular_momenta_in_z_direction(
    positions: &Vec<Vec4>,
    momenta: &Vec<Vec4>,
) -> Vec<Real> {
    assert_eq!(positions.len(), momenta.len());

    positions
        .iter()
        .zip(momenta.iter())
        .map(|(u, v)| u[1] * v[2] - u[2] * v[1])
        .collect()
}
