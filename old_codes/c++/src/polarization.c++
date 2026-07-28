#include "polarization.h++"

using namespace std::complex_literals;

PolarizationVector::PolarizationVector(Complex x, Complex y)
    : x{x}, y{y}
{
    Real norm = std::pow(std::abs(x), 2) + std::pow(std::abs(y), 2);

    if (std::abs(norm - 1) > error_tolerance)
    {
        throw std::invalid_argument("Polarization vector must have norm 1");
    }
}

PolarizationVector::PolarizationVector(Real x, Real y)
    : PolarizationVector(std::complex<Real>{x}, std::complex<Real>{y})
{
}

const PolarizationVector PolarizationVector::linear = PolarizationVector(1, 0);
const PolarizationVector PolarizationVector::right_circular = PolarizationVector(1 / std::sqrt(2), 1i / std::sqrt(2));
const PolarizationVector PolarizationVector::left_circular = PolarizationVector(1 / std::sqrt(2), -1i / std::sqrt(2));
