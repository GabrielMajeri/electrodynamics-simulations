#include "beam.h++"

#include <complex>

#include "constants.h++"

using namespace std::complex_literals;

constexpr Real tolerance = 1e-5;

std::pair<Vector3D, Vector3D> laguerre_gauss_beam_electric_and_magnetic_field(
    Vector3D position, Real time)
{
    const auto [x, y, z] = position;
    const Real r = std::hypot(x, y);
    const Real phi = std::atan2(y, x);

    const Real rayleigh_length = pi * std::pow(waist_radius, 2) / wavelength;

    // w(z)
    const Real width = waist_radius * std::sqrt(1 + std::pow(z / rayleigh_length, 2));

    // r / w(z)
    const Real r_over_width = r / width;
    const Real r_over_width_squared = std::pow(r_over_width, 2);

    // k
    const Real wavenumber = 2 * pi / wavelength;

    // |l|
    const int abs_l = std::abs(azimuthal_index);

    // R(z)
    const Real radius_of_curvature = std::abs(z) < tolerance ? 0 : z * (1 + std::pow(rayleigh_length / z, 2));

    // r^2/(2 * R(z))
    const Real curvature = radius_of_curvature < tolerance ? 0 : std::pow(r, 2) / (2 * radius_of_curvature);

    // \psi(z)
    const auto gouy_phase = std::atan2(z, rayleigh_length);

    std::complex<Real> magnitude = amplitude * (waist_radius / width) * std::pow(std::sqrt(2) * r_over_width, abs_l) * laguerre_polynomial(radial_index, abs_l, 2 * r_over_width_squared) * std::exp(-r_over_width_squared);

    std::complex<Real> phase = std::exp(1i * (angular_velocity * time - wavenumber * z + wavenumber * curvature + azimuthal_index * phi - (2 * radial_index + abs_l + 1) * gouy_phase));

    Complex coeff = magnitude * phase;

    Complex E_x = coeff * polarization.get_x(),
            E_y = coeff * polarization.get_y(),
            E_z = 2i / (wavenumber * std::pow(width, 2)) * (x * E_x + y * E_y);

    Vector3D E = {E_x.real(), E_y.real(), E_z.real()};

    Complex B_x = -E_y / c,
            B_y = E_x / c,
            B_z = 1i / (angular_velocity * std::pow(width, 2)) * (y * E_x - x * E_y);

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
    const auto t = (phi - phi_0) / tau_0;
    return std::exp(-(t * t));
}
