#include "vector.h++"

#include <cmath>

Acceleration operator*(Real scalar, Acceleration acc)
{
    return {scalar * acc.dgamma, scalar * acc.dvx, scalar * acc.dvy, scalar * acc.dvz};
}

Momentum operator+(Momentum m, Acceleration acc)
{
    return {m.gamma + acc.dgamma, m.vx + acc.dvx, m.vy + acc.dvy, m.vz + acc.dvz};
}

Momentum operator*(Real scalar, Momentum m)
{
    return {scalar * m.gamma, scalar * m.vx, scalar * m.vy, scalar * m.vz};
}

Position operator+(Position p, Momentum m)
{
    return {p.t + m.gamma, p.x + m.vx, p.y + m.vy, p.z + m.vz};
}

OPENACC_ROUTINE
Real Vector3D::dot(const Vector3D &rhs) const noexcept
{
    return x * rhs.x + y * rhs.y + z * rhs.z;
}

OPENACC_ROUTINE
Real Vector3D::norm() const noexcept
{
    return std::sqrt(dot(*this));
}

Vector3D Vector3D::normalized() const
{
    return (*this) / norm();
}

Vector3D Vector3D::cross(const Vector3D &rhs) const noexcept
{
    return {
        y * rhs.z - z * rhs.y,
        z * rhs.x - x * rhs.z,
        x * rhs.y - y * rhs.x,
    };
}

Vector3D operator-(Vector3D v, Vector3D w)
{
    return {v.x - w.x, v.y - w.y, v.z - w.z};
}

Vector3D operator*(Real scalar, Vector3D v)
{
    return {scalar * v.x, scalar * v.y, scalar * v.z};
}

Vector3D operator*(Vector3D v, Real scalar)
{
    return scalar * v;
}

Vector3D operator/(Vector3D v, Real scalar)
{
    return {v.x / scalar, v.y / scalar, v.z / scalar};
}

OPENACC_ROUTINE
ComplexVector3D &ComplexVector3D::operator+=(ComplexVector3D v) noexcept
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

ComplexVector3D operator+(ComplexVector3D v, ComplexVector3D w)
{
    return {v.x + w.x, v.y + w.y, v.z + w.z};
}

ComplexVector3D operator-(ComplexVector3D v, ComplexVector3D w)
{
    return {v.x - w.x, v.y - w.y, v.z - w.z};
}

ComplexVector3D operator*(Complex scalar, ComplexVector3D v)
{
    return {scalar * v.x, scalar * v.y, scalar * v.z};
}
