#include <cassert>
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENACC
#include <openacc.h>
#define OPENACC_ROUTINE _Pragma("acc routine seq")
#else
#define OPENACC_ROUTINE
#endif

using Real = double;

/// The mathematical constant $\pi$.
constexpr Real pi = std::numbers::pi;

/// Speed of light in vacuum (in atomic units it's equal to the fine structure constant).
constexpr Real c = 137.036;
#ifdef _OPENACC
#pragma acc declare copyin(c)
#endif

/// Set to `true` to enable checking for errors in numerical integration code.
/// (e.g. check that energy conservation holds, ensure the Lorentz factor is always >= 1,
/// ensure orthogonality of velocity and acceleration vectors)
constexpr bool check_for_errors = false;

/// Tolerance for numerical errors.
constexpr Real error_tolerance = 1e-5;

/// @brief Position 4-vector. Identifies an event (point) in spacetime.
struct Position
{
    Real t, x, y, z;
};

/// @brief Momentum 4-vector. Contains the relativistic factor and the momentum of the particle.
struct Momentum
{
    Real gamma, vx, vy, vz;
};

/// @brief Acceleration 4-vector.
struct Acceleration
{
    Real dgamma, dvx, dvy, dvz;
};

OPENACC_ROUTINE
static Acceleration operator*(Real scalar, Acceleration acc)
{
    return Acceleration{scalar * acc.dgamma, scalar * acc.dvx, scalar * acc.dvy, scalar * acc.dvz};
}

OPENACC_ROUTINE
static Momentum operator+(Momentum m, Acceleration acc)
{
    return Momentum{m.gamma + acc.dgamma, m.vx + acc.dvx, m.vy + acc.dvy, m.vz + acc.dvz};
}

OPENACC_ROUTINE
static Momentum operator*(Real scalar, Momentum m)
{
    return Momentum{scalar * m.gamma, scalar * m.vx, scalar * m.vy, scalar * m.vz};
}

OPENACC_ROUTINE
static Position operator+(Position p, Momentum m)
{
    return Position{p.t + m.gamma, p.x + m.vx, p.y + m.vy, p.z + m.vz};
}

/// @brief Position vector in 3D Euclidean space.
struct Vector3D
{
    Real x, y, z;

    static Vector3D from_position(const Position &position)
    {
        return Vector3D(position.x, position.y, position.z);
    }
};

OPENACC_ROUTINE
static Vector3D operator*(Real scalar, Vector3D v)
{
    return Vector3D{scalar * v.x, scalar * v.y, scalar * v.z};
}

/// @brief Represents one of the possible polarizations of the EM field.
class PolarizationVector
{
    std::complex<Real> x, y;

public:
    PolarizationVector(std::complex<Real> x, std::complex<Real> y)
        : x{x}, y{y}
    {
        Real norm = std::pow(std::abs(x), 2) + std::pow(std::abs(y), 2);

        if (std::abs(norm - 1) > error_tolerance)
        {
            throw std::invalid_argument("Polarization vector must have norm 1");
        }
    }

    PolarizationVector(Real x, Real y)
        : PolarizationVector(std::complex<Real>{x}, std::complex<Real>{y})
    {
    }

    const std::complex<Real> get_x() const noexcept
    {
        return x;
    }
    const std::complex<Real> get_y() const noexcept
    {
        return y;
    }

    static const PolarizationVector linear, right_circular, left_circular;
};

using namespace std::complex_literals;

const PolarizationVector PolarizationVector::linear = PolarizationVector(1, 0);
const PolarizationVector PolarizationVector::right_circular = PolarizationVector(1 / std::sqrt(2), 1i / std::sqrt(2));
const PolarizationVector PolarizationVector::left_circular = PolarizationVector(1 / std::sqrt(2), -1i / std::sqrt(2));

/// @brief Parameters controlling the shape and time evolution of a Laguerre-Gauss beam.
struct LaguerreGaussBeamParameters
{
    // E_0 = B_0
    Real amplitude;
    // \xi
    PolarizationVector polarization;
    // w(0)
    Real waist_radius;
    // \lambda
    Real wavelength;
    // \omega
    Real angular_velocity;
    // p
    uint32_t radial_index;
    // l (or m)
    int32_t azimuthal_index;
};

static std::vector<Position> generate_initial_electron_positions(std::size_t num_electrons, double disk_radius, uint32_t seed);
static std::vector<Momentum> generate_initial_electron_momenta(std::size_t num_electrons);
static std::pair<std::vector<Position>, std::vector<Momentum>> integrate_trajectories(
    LaguerreGaussBeamParameters parameters,
    Real charge_to_mass_ratio,
    Real phi_0, Real tau_0,
    std::vector<Position> initial_positions,
    std::vector<Momentum> initial_momenta,
    Real integration_start_time, Real integration_end_time,
    Real time_step);

OPENACC_ROUTINE
static std::pair<Vector3D, Vector3D> laguerre_gauss_beam_electric_and_magnetic_field(
    LaguerreGaussBeamParameters parameters, Vector3D position, Real time);

OPENACC_ROUTINE
static Real laguerre_polynomial(uint32_t n, Real alpha, Real x);

OPENACC_ROUTINE
static Real cutoff(Real phi, Real phi_0, Real tau_0);

OPENACC_ROUTINE
static Acceleration compute_acceleration(
    Momentum previous_momentum, Real charge_to_mass_ratio,
    Vector3D electric_field, Vector3D magnetic_field);

static std::vector<Real> compute_angular_momenta_in_z_direction(
    Real particle_mass,
    std::vector<Position> positions, std::vector<Momentum> momenta);

template <typename T>
static void write_npy_file(const char file_path[], const std::vector<T> &array);

static std::vector<char> &operator+=(std::vector<char> &v, const char *s)
{
    while (*s)
    {
        v.push_back(*s);
        ++s;
    }

    return v;
}

int main()
{
    std::cout << "Starting Laguerre-Gauss beam angular momentum transfer simulation code" << std::endl;

#ifdef _OPENMP
    std::cout << "Using OpenMP with " << omp_get_thread_limit() << " threads" << std::endl;
#else
#ifdef _OPENACC
    std::cout << "Using OpenACC\n";

    const auto device_type = acc_get_device_type();
    std::cout << "OpenACC device type: ";
    if (device_type == acc_device_host)
    {
        std::cout << "Host (CPU)";
    }
    else if (device_type == acc_device_nvidia)
    {
        std::cout << "NVIDIA";
    }
    else
    {
        std::cout << device_type;
    }
    std::cout << std::endl;

#else
    std::cout << "Warning: not using any sort of accelerator or parallelization" << std::endl;
#endif
#endif

    // constexpr size_t num_electrons = 2 * 1024;
    // constexpr size_t num_electrons = 16 * 1024;
    constexpr size_t num_electrons = 64 * 1024;

    constexpr Real omega = 0.057;
    constexpr Real lambda = 2 * pi * c / omega;
    constexpr Real waist_radius = 75 * lambda;

    // TODO: fix convergence issues for larger values of a_0
    constexpr Real a_0 = 1e-2;
    // constexpr Real a_0 = 1e-0;
    constexpr Real m_e = 1, q = -1;
    constexpr Real charge_to_mass_ratio = q / m_e;

    constexpr Real amplitude = a_0 * m_e * c * omega / std::abs(q);
    // const PolarizationVector polarization = PolarizationVector::linear;
    const PolarizationVector polarization = PolarizationVector::right_circular;

    constexpr uint32_t radial_index = 2;
    constexpr int32_t azimuthal_index = -2;

    constexpr Real disk_radius = (1.75 + radial_index) * waist_radius;

    constexpr uint32_t seed = 42;

    std::cout << "Generating initial positions for " << num_electrons << " electrons, uniformly distributed within a disk of radius " << disk_radius << " in the x-y plane, centered at the origin" << std::endl;

    auto start = std::chrono::steady_clock::now();

    const auto initial_electron_positions = generate_initial_electron_positions(num_electrons, disk_radius, seed);

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = finish - start;

    std::cout << "Generated " << num_electrons << " initial positions in " << elapsed_seconds.count() << " seconds" << std::endl;

    std::cout << "Writing initial electron positions to disk..." << std::endl;
    write_npy_file("initial_positions.npy", initial_electron_positions);

    const auto initial_electron_momenta = generate_initial_electron_momenta(num_electrons);

    const LaguerreGaussBeamParameters beam_parameters{
        amplitude, polarization, waist_radius, lambda,
        omega, radial_index, azimuthal_index};

    constexpr auto tau_0 = 10 / omega;
    constexpr auto phi_0 = 3 * tau_0;

    constexpr Real start_time = 0.0, end_time = 6 * tau_0;
    constexpr Real time_step = tau_0 / 2048;
    std::cout << "Integrating equations of motions from t_0 = " << start_time << " up to t_final = " << end_time << ", with a time step of " << time_step << std::endl;

    start = std::chrono::steady_clock::now();

    const auto [final_positions, final_momenta] =
        integrate_trajectories(
            beam_parameters, charge_to_mass_ratio, phi_0, tau_0, initial_electron_positions,
            initial_electron_momenta, start_time, end_time, time_step);

    finish = std::chrono::steady_clock::now();
    elapsed_seconds = finish - start;

    std::cout << "Integrating " << num_electrons << " trajectories took " << elapsed_seconds.count() << " seconds" << std::endl;

#ifdef _OPENACC
    // BUGFIX: if I don't shutdown OpenACC explicitly here, it crashes (returns a non-zero exit code) at program exit
    acc_shutdown(acc_get_device_type());
#endif

    std::cout << "Computing angular momentum in the z direction for electrons in the final state" << std::endl;

    start = std::chrono::steady_clock::now();

    const auto angular_momenta = compute_angular_momenta_in_z_direction(
        m_e, final_positions, final_momenta);

    finish = std::chrono::steady_clock::now();
    elapsed_seconds = finish - start;

    std::cout << "Computing angular momenta for " << num_electrons << " electrons took " << elapsed_seconds.count() << " seconds" << std::endl;

    std::cout << "Writing angular momenta to disk..." << std::endl;
    write_npy_file("angular_momenta.npy", angular_momenta);

    std::cout << "Done" << std::endl;

    return 0;
}

std::vector<Position> generate_initial_electron_positions(size_t num_electrons, double disk_radius, uint32_t seed)
{
    std::vector<Position> positions(num_electrons);

    // Use the inverse sampling method to generate points uniformly in the disk
    const double disk_radius_squared = disk_radius * disk_radius;

    std::uniform_real_distribution<Real>
        unif_r(0.0, disk_radius_squared),
        unif_angle(0.0, 2 * pi);

    std::default_random_engine rng(seed);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        // Generate a new position for each electron
        const double r = std::sqrt(unif_r(rng));
        const double theta = unif_angle(rng);

        const Real
            x = r * std::cos(theta),
            y = r * std::sin(theta),
            z = 0;

        positions[i] = Position{0, x, y, z};
    }

    return positions;
}

std::vector<Momentum> generate_initial_electron_momenta(size_t num_electrons)
{
    std::vector<Momentum> momenta(num_electrons);

    for (size_t i = 0; i < num_electrons; ++i)
    {
        momenta[i].gamma = 1;
    }

    return momenta;
}

std::pair<std::vector<Position>, std::vector<Momentum>> integrate_trajectories(
    LaguerreGaussBeamParameters parameters, Real charge_to_mass_ratio,
    Real phi_0, Real tau_0,
    std::vector<Position> initial_positions, std::vector<Momentum> initial_momenta,
    Real integration_start_time, Real integration_end_time,
    Real time_step)
{
    // Determine number of particles
    const size_t num_particles = initial_positions.size();
    assert(num_particles == initial_momenta.size());

    // Allocate some buffers to store positions/momenta during integration
    std::vector<Position> positions = initial_positions;
    std::vector<Momentum> momenta = initial_momenta;

    const Real integration_duration = integration_end_time - integration_start_time;
    const size_t num_steps = integration_duration / time_step;

    Position *positions_arr = positions.data();
    Momentum *momenta_arr = momenta.data();

#ifdef _OPENMP
#pragma omp parallel for
#else
#ifdef _OPENACC
#pragma acc parallel loop copy(positions_arr[ : num_particles], momenta_arr[ : num_particles])
#endif
#endif
    for (size_t index = 0; index < num_particles; ++index)
    {
        Real current_time = 0;

        for (size_t step = 0; step <= num_steps; ++step)
        {
            const auto previous_position = positions_arr[index];
            const auto laboratory_time = previous_position.t;
            const auto position_vector = Vector3D::from_position(previous_position);

            // Compute EM field vectors for previous position
            auto [electric_field, magnetic_field] =
                laguerre_gauss_beam_electric_and_magnetic_field(parameters, position_vector, laboratory_time);

            const auto cf = cutoff(laboratory_time - previous_position.z / c, phi_0, tau_0);
            electric_field = cf * electric_field;
            magnetic_field = cf * magnetic_field;

            const auto previous_momentum = momenta_arr[index];

            // Symplectic Euler integration step
            const auto acceleration = compute_acceleration(previous_momentum, charge_to_mass_ratio, electric_field, magnetic_field);
            const auto new_momentum = previous_momentum + time_step * acceleration;
            const auto new_position = previous_position + time_step * new_momentum;

            if (check_for_errors)
            {
                if (new_momentum.gamma < 1 - error_tolerance)
                {
                    std::cout << "Lorentz factor dropped below unity: " << new_momentum.gamma << std::endl;
                    std::exit(1);
                }

                const auto inner_product = acceleration.dvx * previous_momentum.vx + acceleration.dvy * previous_momentum.vy + acceleration.dvz * previous_momentum.vz - acceleration.dgamma * previous_momentum.gamma;

                if (std::abs(inner_product) > error_tolerance)
                {
                    std::cout << "Inner product is non-zero: " << std::abs(inner_product) << std::endl;
                    std::exit(1);
                }
            }

            positions_arr[index] = new_position;
            momenta_arr[index] = new_momentum;
        }

        current_time += time_step;
    }

    return std::make_pair(positions, momenta);
}

std::pair<Vector3D, Vector3D> laguerre_gauss_beam_electric_and_magnetic_field(
    LaguerreGaussBeamParameters parameters,
    Vector3D position, Real time)
{
    const Real r = std::hypot(position.x, position.y);
    const Real phi = std::atan2(position.y, position.x);
    const auto [x, y, z] = position;

    const Real rayleigh_length = pi * std::pow(parameters.waist_radius, 2) / parameters.wavelength;

    // w(z)
    const Real width = parameters.waist_radius * std::sqrt(1 + std::pow(z / rayleigh_length, 2));

    // r / w(z)
    const Real r_over_width = r / width;
    const Real r_over_width_squared = std::pow(r_over_width, 2);

    // k
    const Real wavenumber = 2 * pi / parameters.wavelength;

    // |l|
    const int abs_l = std::abs(parameters.azimuthal_index);

    // R(z)
    const Real radius_of_curvature = std::abs(z) < 1e-5 ? 0 : z * (1 + std::pow(rayleigh_length / z, 2));

    // r^2/(2 * R(z))
    const Real curvature = radius_of_curvature == 0 ? 0 : std::pow(r, 2) / (2 * radius_of_curvature);

    // \psi(z)
    const auto gouy_phase = std::atan2(z, rayleigh_length);

    std::complex<Real> magnitude = parameters.amplitude * (parameters.waist_radius / width) * std::pow(std::sqrt(2) * r_over_width, abs_l) * laguerre_polynomial(parameters.radial_index, abs_l, 2 * r_over_width_squared) * std::exp(-r_over_width_squared);

    std::complex<Real> phase = std::exp(1i * parameters.angular_velocity * time - 1i * (wavenumber * z + wavenumber * curvature + parameters.azimuthal_index * phi - (2 * parameters.radial_index + abs_l + 1) * gouy_phase));

    std::complex<Real> coeff = magnitude * phase;

    std::complex<Real> E_x = coeff * parameters.polarization.get_x(),
                       E_y = coeff * parameters.polarization.get_y(),
                       E_z = 2i / (wavenumber * std::pow(width, 2)) * (x * E_x + y * E_y);

    Vector3D E = {E_x.real(), E_y.real(), E_z.real()};

    std::complex<Real> B_x = -E_y / c,
                       B_y = E_x / c,
                       B_z = 1i / (parameters.angular_velocity * std::pow(width, 2)) * (y * E_x - x * E_y);

    Vector3D B = {B_x.real(), B_y.real(), B_z.real()};

    return std::make_pair(E, B);
}

// Laguerre gauss polynomial evaluation (fast for small n).
Real laguerre_polynomial(uint32_t n, Real alpha, Real x)
{
    if (n == 0)
    {
        return 1;
    }
    if (n == 1)
    {
        return 1 + alpha - x;
    }
    if (n == 2)
    {
        return 0.5 * (std::pow(x, 2) - 2 * (alpha + 2) * x + (alpha + 1) * (alpha + 2));
    }

    return ((2 * n - 1 + alpha - x) * laguerre_polynomial(n - 1, alpha, x) - (n - 1 + alpha) * laguerre_polynomial(n - 2, alpha, x)) / n;
}

Real cutoff(Real phi, Real phi_0, Real tau_0)
{
    return std::exp(-std::pow((phi - phi_0) / tau_0, 2));
}

Acceleration compute_acceleration(Momentum previous_momentum, Real charge_to_mass_ratio, Vector3D electric_field, Vector3D magnetic_field)
{
    const auto agamma = previous_momentum.vx * electric_field.x / c + previous_momentum.vy * electric_field.y / c + previous_momentum.vz * electric_field.z / c;
    const auto ax = previous_momentum.gamma * electric_field.x / c + previous_momentum.vy * magnetic_field.z - previous_momentum.vz * magnetic_field.y;
    const auto ay = previous_momentum.gamma * electric_field.y / c - previous_momentum.vx * magnetic_field.z + previous_momentum.vz * magnetic_field.x;
    const auto az = previous_momentum.gamma * electric_field.z / c - previous_momentum.vx * magnetic_field.y - previous_momentum.vy * magnetic_field.x;

    const Acceleration acceleration_direction{agamma, ax, ay, az};
    return charge_to_mass_ratio * acceleration_direction;
}

std::vector<Real> compute_angular_momenta_in_z_direction(
    Real particle_mass,
    std::vector<Position> positions, std::vector<Momentum> momenta)
{
    std::size_t num_particles = positions.size();
    std::vector<Real> angular_momenta(num_particles);

    for (std::size_t index = 0; index < num_particles; ++index)
    {
        const auto position = positions[index];
        const auto momentum = momenta[index];

        angular_momenta[index] = particle_mass * (position.x * momentum.vy - position.y * momentum.vx);
    }

    return angular_momenta;
}

template <typename T>
void write_npy_file(const char *file_path, const std::vector<T> &array)
{
    std::ofstream output(file_path, std::ios::binary);
    output.exceptions(std::ostream::badbit);

    // Start constructing the header in a memory buffer
    std::vector<uint8_t> header;
    header.reserve(64);

    // Magic number
    header.push_back(0x93);
    const char numpyString[] = "NUMPY";
    for (const char *pc = numpyString; *pc != 0; ++pc)
    {
        header.push_back(*pc);
    }

    // Version 3.0
    header.push_back(0x03);
    header.push_back(0x00);

    output.write(reinterpret_cast<const char *>(header.data()), header.size());

    // Array metadata dictionary
    std::vector<char> dictionary;
    dictionary.reserve(128);

    std::stringstream ss;
    ss << "<f";
    ss << sizeof(Real);

    dictionary += "{'descr':";
    dictionary.push_back('\'');
    dictionary += ss.str().data();
    dictionary.push_back('\'');

    dictionary.push_back(',');
    dictionary += "'fortran_order':False";

    dictionary.push_back(',');
    dictionary += "'shape':";

    ss.str("");
    ss.clear();

    ss << '(' << array.size() << ',' << (sizeof(T) / sizeof(Real)) << ')';

    dictionary += ss.str().data();

    dictionary += "}";

    const auto fixed_header_length = 6 + 2 + 4;
    auto dictionary_length = dictionary.size();
    do
    {
        dictionary.push_back(' ');
        ++dictionary_length;
    } while ((fixed_header_length + dictionary_length) % 16 != 0);

    dictionary.back() = '\n';

    // Header dictionary length
    uint32_t dict_length = dictionary.size();

    output.write(reinterpret_cast<const char *>(&dict_length), 4);

    // Header dictionary contents
    output.write(dictionary.data(), dictionary.size());

    output.write(reinterpret_cast<const char *>(array.data()), array.size() * sizeof(T));

    output.close();
}
