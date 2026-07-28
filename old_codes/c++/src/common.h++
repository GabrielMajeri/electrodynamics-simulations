#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENACC
#include <openacc.h>
#define OPENACC_ROUTINE _Pragma("acc routine seq")
#else
#define OPENACC_ROUTINE
#endif

/// Set to `true` to enable checking for errors in numerical integration code.
/// (e.g. check that energy conservation holds, ensure the Lorentz factor is always >= 1,
/// ensure orthogonality of velocity and acceleration vectors)
constexpr bool check_for_errors = false;

/// Tolerance for numerical errors.
constexpr double error_tolerance = 1e-5;
