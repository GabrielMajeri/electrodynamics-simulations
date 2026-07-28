#pragma once

#include "common.h++"
#include "types.h++"

/// @brief Represents one of the possible polarizations of the EM field.
class PolarizationVector
{
    Complex x, y;

public:
    PolarizationVector(Complex x, Complex y);

    PolarizationVector(Real x, Real y);

    inline const std::complex<Real> get_x() const noexcept
    {
        return x;
    }
    inline const std::complex<Real> get_y() const noexcept
    {
        return y;
    }

    static const PolarizationVector linear, right_circular, left_circular;
};
