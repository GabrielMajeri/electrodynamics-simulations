use nalgebra::vector;

use crate::{
    beam::laguerre_gauss_beam_electric_and_magnetic_field,
    constants::{
        CHARGE_TO_MASS_RATIO, INTEGRATION_DURATION, INTEGRATION_END_TIME, INTEGRATION_START_TIME,
        INTEGRATION_TIME_STEP, NUM_INTEGRATION_STEPS, NUM_PARTICLES, PHI_0, SPEED_OF_LIGHT, TAU_0,
    },
    initial_conditions::InitialConditions,
    types::{Real, Vec3, Vec4},
};

pub struct IntegrationResult {
    pub positions: Vec<Vec4>,
    pub momenta: Vec<Vec4>,
}

/// Integrate the given initial conditions for the integration duration and
/// return the final positions and momenta.
pub fn integrate_trajectories(initial_conditions: InitialConditions) -> IntegrationResult {
    let InitialConditions {
        mut positions,
        mut momenta,
    } = initial_conditions;

    assert_eq!(positions.len(), momenta.len());

    println!(
        "Beginning to integrate trajectories for {} time steps, from {} to {}",
        INTEGRATION_DURATION, INTEGRATION_START_TIME, INTEGRATION_END_TIME
    );

    for particle_index in 0..NUM_PARTICLES {
        for _integration_step in 0..NUM_INTEGRATION_STEPS {
            let previous_position = positions[particle_index];
            let laboratory_time = previous_position.x;
            let position_vector = vector![
                previous_position.y,
                previous_position.z,
                previous_position.w
            ];

            // Compute EM field vectors for previous position
            let (electric_field, magnetic_field) =
                laguerre_gauss_beam_electric_and_magnetic_field(position_vector, laboratory_time);

            let coeff = temporal_cut_off(laboratory_time - position_vector.z / SPEED_OF_LIGHT);

            let electric_field = coeff * electric_field;
            let magnetic_field = coeff * magnetic_field;

            let previous_momentum = momenta[particle_index];

            // Euler integration step
            let acceleration =
                compute_acceleration(previous_momentum, electric_field, magnetic_field);

            let new_momentum = previous_momentum + INTEGRATION_TIME_STEP * acceleration;
            let new_position = previous_position + INTEGRATION_TIME_STEP * new_momentum;

            positions[particle_index] = new_position;
            momenta[particle_index] = new_momentum;
        }
    }

    IntegrationResult { positions, momenta }
}

/// Laser beam envelope (Gaussian shape)
fn temporal_cut_off(phi: Real) -> Real {
    let t = (phi - PHI_0) / TAU_0;
    let exponent = -(t * t);
    return exponent.exp();
}

/// Compute the acceleration vector, by multiplying the EM field tensor
/// with the Minkowski metric tensor and the previous momentum.
fn compute_acceleration(
    previous_momentum: Vec4,
    electric_field: Vec3,
    magnetic_field: Vec3,
) -> Vec4 {
    let gamma = previous_momentum[1] * electric_field.x / SPEED_OF_LIGHT
        + previous_momentum[2] * electric_field.y / SPEED_OF_LIGHT
        + previous_momentum[3] * electric_field.z / SPEED_OF_LIGHT;
    let x = previous_momentum[0] * electric_field.x / SPEED_OF_LIGHT
        + previous_momentum[2] * magnetic_field.z
        - previous_momentum[3] * magnetic_field.y;
    let y = previous_momentum[0] * electric_field.y / SPEED_OF_LIGHT
        - previous_momentum[1] * magnetic_field.z
        + previous_momentum[3] * magnetic_field.x;
    let z = previous_momentum[0] * electric_field.z / SPEED_OF_LIGHT
        + previous_momentum[1] * magnetic_field.y
        - previous_momentum[2] * magnetic_field.x;

    return CHARGE_TO_MASS_RATIO * vector![gamma, x, y, z];
}
