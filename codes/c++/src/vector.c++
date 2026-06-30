#include "vector.h++"

#include <cmath>
#include <iostream>

Acceleration operator+(Acceleration a1, Acceleration a2)
{
    return {a1.du0 + a2.du0, a1.du1 + a2.du1, a1.du2 + a2.du2, a1.du3 + a2.du3};
}

Acceleration operator*(Real scalar, Acceleration acc)
{
    return {scalar * acc.du0, scalar * acc.du1, scalar * acc.du2, scalar * acc.du3};
}

Momentum operator+(Momentum m, Acceleration acc)
{
    return {m.u0 + acc.du0, m.u1 + acc.du1, m.u2 + acc.du2, m.u3 + acc.du3};
}

Momentum operator*(Real scalar, Momentum m)
{
    return {scalar * m.u0, scalar * m.u1, scalar * m.u2, scalar * m.u3};
}

std::ostream &operator<<(std::ostream &out, Momentum m)
{
    out << m.u0 << ' ' << m.u1 << ' ' << m.u2 << ' ' << m.u3;

    return out;
}

Position operator+(Position p, Momentum m)
{
    return {p.t + m.u0, p.x + m.u1, p.y + m.u2, p.z + m.u3};
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

Vector3D operator+(Vector3D v, Vector3D w)
{
    return {v.x + w.x, v.y + w.y, v.z + w.z};
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

ComplexVector3D operator/(ComplexVector3D v, Complex scalar)
{
    return {v.x / scalar, v.y / scalar, v.z / scalar};
}
