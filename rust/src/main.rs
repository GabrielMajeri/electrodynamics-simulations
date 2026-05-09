use std::time::Instant;

use nalgebra::vector;

use crate::{
    angular_momentum::compute_angular_momenta_in_z_direction,
    constants::{DISK_RADIUS, NUM_PARTICLES},
    integrate::integrate_trajectories,
    numpy::write_to_npy_file,
};

mod angular_momentum;
mod beam;
mod constants;
mod initial_conditions;
mod integrate;
mod numpy;
mod types;

fn main() {
    println!("Starting Laguerre-Gauss beam angular momentum transfer simulation code");

    println!(
        "Generating initial conditions for {} electrons, uniformly distributed within the circle of radius {} in the x-y plane (at z = 0)",
        NUM_PARTICLES, DISK_RADIUS
    );

    let start_time = Instant::now();

    let initial_conditions = initial_conditions::generate_initial_conditions();

    let duration = start_time.elapsed();

    println!(
        "Generated {} initial conditions (4-positions and 4-momenta) in {} seconds",
        NUM_PARTICLES,
        duration.as_secs_f64()
    );

    println!("Writing initial positions to disk...");
    let start_time = Instant::now();

    write_to_npy_file("initial_positions.npy", &initial_conditions.positions);

    let duration = start_time.elapsed();
    println!(
        "Took {} seconds to save initial positions to disk",
        duration.as_secs_f64()
    );

    let start_time = Instant::now();

    let integration_result = integrate_trajectories(initial_conditions);

    let duration = start_time.elapsed();

    println!(
        "Integrating {} trajectories took {} seconds",
        NUM_PARTICLES,
        duration.as_secs_f64()
    );

    let angular_momenta = compute_angular_momenta_in_z_direction(
        &integration_result.positions,
        &integration_result.momenta,
    );
    let angular_momenta = angular_momenta.into_iter().map(|am| vector![am]).collect();

    println!("Writing angular momenta to disk...");
    let start_time = Instant::now();

    write_to_npy_file("angular_momenta.npy", &angular_momenta);

    let duration = start_time.elapsed();
    println!(
        "Took {} seconds to save final angular momenta to disk",
        duration.as_secs_f64()
    );
}
