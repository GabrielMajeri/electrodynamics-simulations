module ElectrodynamicsSimulations

using Base.Threads: @threads
using IOCapture: IOCapture
using LaserTypes
using LinearAlgebra: dot
using NPZ: npzwrite
using Printf
import Random
using StaticArrays

const c = 137.036
const NUM_ELECTRONS = 64 * 1024

const MINKOWSKI_METRIC_TENSOR = @SMatrix [
    1 0 0 0;
    0 -1 0 0;
    0 0 -1 0;
    0 0 0 -1
]

const ω = 0.057
const λ = 2 * π * c / ω
const WAIST_RADIUS = 75 * λ

# TODO: fix convergence issues for a₀ >= 1e-1
const a₀ = 1e-2
const mₑ = 1
const q = -1

const AMPLITUDE = a₀ * mₑ * c * ω * abs(q)

const CHARGE_TO_MASS_RATIO = q / mₑ

struct PolarizationVector
    x::ComplexF64
    y::ComplexF64
end

const POLARIZATION_LINEAR = PolarizationVector(1, 0)
const POLARIZATION_RIGHT_CIRCULAR = PolarizationVector(1 / sqrt(2), 1im / sqrt(2))
const POLARIZATION_LEFT_CIRCULAR = PolarizationVector(1 / sqrt(2), -1im / sqrt(2))

const POLARIZATION = POLARIZATION_RIGHT_CIRCULAR

const RADIAL_INDEX = 2
const AZIMUTHAL_INDEX = -2

const DISK_RADIUS = (1.75 + RADIAL_INDEX) * WAIST_RADIUS

const τ₀ = 10 / ω
const φ₀ = 3 * τ₀

const INTEGRATION_START_TIME = 0
const INTEGRATION_END_TIME = 6 * τ₀
const TIME_STEP = τ₀ / 2048

const ERROR_TOLERANCE = 1e-8

const RealType = Float64
const Vec3 = SVector{3,RealType}
const Vec4 = SVector{4,RealType}

function generate_initial_positions(num_particles)
    positions = Vector{Vec4}(undef, num_particles)

    disk_radius_squared = DISK_RADIUS^2

    for index = 1:num_particles
        u1, u2 = rand(2)

        r = sqrt(u1 * disk_radius_squared)
        theta = 2 * π * u2

        sine, cosine = sincos(theta)

        x = r * cosine
        y = r * sine
        z = 0
        positions[index] = @SVector [0, x, y, z]
    end

    positions
end

function generate_initial_momenta(num_particles)
    momenta = Vector{Vec4}(undef, num_particles)

    for index = 1:num_particles
        momenta[index] = @SVector [1, 0, 0, 0]
    end

    momenta
end

function generate_initial_conditions(num_particles)
    @printf "Generating initial conditions for %d electrons\n" num_particles

    Random.seed!(1234)

    println("Generating initial electron positions")
    initial_positions = generate_initial_positions(num_particles)

    println("Generating initial electron momenta")
    initial_momenta = generate_initial_momenta(num_particles)

    initial_positions, initial_momenta
end

function convert_vector_of_svector_to_array(matrix)
    result = Array{RealType}(undef, length(matrix), length(matrix[1]))

    for index = eachindex(matrix)
        result[index, :] = matrix[index]
    end

    result
end

function integrate_trajectories(positions, momenta)
    num_particles = length(positions)
    @assert length(momenta) == num_particles

    println("Setting up laser with Laguerre-Gauss beam")
    laser = setup_laser(LaguerreGaussLaser, :atomic;
        λ=λ, w₀=WAIST_RADIUS, a₀=a₀, τ=τ₀, ξx=POLARIZATION.x, ξy=POLARIZATION.y)

    integration_duration = INTEGRATION_END_TIME - INTEGRATION_START_TIME
    num_steps = integration_duration / TIME_STEP

    println("Starting numerical integration using symplectic Euler method")

    @printf "Integrating trajectories from t_initial = %f to t_final = %f, with a time step of %f\n" INTEGRATION_START_TIME INTEGRATION_END_TIME TIME_STEP

    @threads :static for particle_index = 1:num_particles
        for step = 1:num_steps
            previous_position = positions[particle_index]
            previous_momentum = momenta[particle_index]

            em_tensor = Fμν(previous_position, laser)
            previous_momentum_lower_indices = MINKOWSKI_METRIC_TENSOR * previous_momentum
            acceleration = em_tensor * previous_momentum_lower_indices

            new_momentum = previous_momentum + TIME_STEP * acceleration
            new_position = previous_position + TIME_STEP * new_momentum

            # @assert new_momentum[1] >= 1 - ERROR_TOLERANCE
            # @assert isapprox(dot(acceleration, previous_momentum_lower_indices), 0; atol=ERROR_TOLERANCE)

            positions[particle_index] = new_position
            momenta[particle_index] = new_momentum
        end
    end

    positions, momenta
end

function compute_angular_momenta(positions, momenta)
    num_particles = length(positions)
    @assert length(momenta) == num_particles

    angular_momenta = Vector{Float64}(undef, num_particles)

    for index = 1:num_particles
        position = positions[index]
        momentum = momenta[index]

        angular_momenta[index] = mₑ * (position[2] * momentum[3] - position[3] * momentum[2])
    end

    angular_momenta
end

@printf "Available threads: %d\n" Threads.nthreads()

println("Precompiling/warming up code...")
IOCapture.capture() do
    initial_positions, initial_momenta = generate_initial_conditions(4)
    final_positions, final_momenta = integrate_trajectories(initial_positions, initial_momenta)
    compute_angular_momenta(final_positions, final_momenta)
end

println("Generating initial conditions for electrons bunch...")
@time initial_positions, initial_momenta = generate_initial_conditions(NUM_ELECTRONS)

npzwrite("initial_positions.npy", convert_vector_of_svector_to_array(initial_positions))

println("Integrating electron trajectories under laser pulse...")
@time final_positions, final_momenta = integrate_trajectories(initial_positions, initial_momenta)

println("Computing angular momentum in the z direction for each electron...")
@time angular_momenta = compute_angular_momenta(final_positions, final_momenta)

npzwrite("angular_momenta.npy", angular_momenta)

end # module ElectrodynamicsSimulations
