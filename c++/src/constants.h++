#pragma once

#include <cstdint>
#include <numbers>

#include "polarization.h++"
#include "types.h++"

/// The mathematical constant $\pi$.
constexpr Real pi = std::numbers::pi;

/// Speed of light in vacuum (in atomic units it's equal to the fine structure constant).
constexpr Real c = 137.036;
#ifdef _OPENACC
#pragma acc declare copyin(c)
#endif

constexpr size_t num_electrons = 4 * 1024;
// constexpr size_t num_electrons = 16 * 1024;
// constexpr std::size_t num_electrons = 64 * 1024;
// constexpr std::size_t num_electrons = 128 * 1024;

constexpr Real omega = 0.057;
constexpr Real angular_velocity = omega;
constexpr Real lambda = 2 * pi * c / omega;
constexpr Real wavelength = lambda;
constexpr Real waist_radius = 75 * lambda;

// TODO: fix convergence issues for larger values of a_0
constexpr Real a_0 = 1e-2;
// constexpr Real a_0 = 1e-0;
constexpr Real m_e = 1, q = -1;
constexpr Real charge_to_mass_ratio = q / m_e;

constexpr Real amplitude = a_0 * m_e * c * omega / std::abs(q);
// const PolarizationVector polarization = PolarizationVector::linear;
// const PolarizationVector polarization = PolarizationVector::right_circular;
const PolarizationVector polarization = PolarizationVector::left_circular;

constexpr std::uint32_t radial_index = 2;
constexpr std::int32_t azimuthal_index = -2;

constexpr Real disk_radius = (1.75 + radial_index) * waist_radius;

constexpr auto tau_0 = 10 / omega;
constexpr auto phi_0 = 3 * tau_0;

constexpr Real integration_start_time = 0.0, integration_end_time = 6 * tau_0;
constexpr Real integration_time_step = 1e-1;

constexpr std::size_t detector_grid_size_x = 64;
constexpr std::size_t detector_grid_size_y = 64;
constexpr std::size_t num_detector_points = detector_grid_size_x * detector_grid_size_y;
constexpr Real detector_width = 2500 * lambda;
constexpr Real detector_height = 2500 * lambda;
