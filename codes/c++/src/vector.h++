#pragma once

#include "common.h++"
#include "types.h++"

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
Acceleration operator*(Real scalar, Acceleration acc);

OPENACC_ROUTINE
Momentum operator+(Momentum m, Acceleration acc);

OPENACC_ROUTINE
Momentum operator*(Real scalar, Momentum m);

OPENACC_ROUTINE
Position operator+(Position p, Momentum m);

/// @brief Position vector in 3D Euclidean space.
struct Vector3D
{
    Real x, y, z;

    static inline Vector3D from_position(const Position &position)
    {
        return {position.x, position.y, position.z};
    }

    static inline Vector3D from_momentum(const Momentum &momentum)
    {
        return {momentum.vx, momentum.vy, momentum.vz};
    }

    Real dot(const Vector3D &rhs) const noexcept;
    Real norm() const noexcept;
    Vector3D normalized() const;

    Vector3D cross(const Vector3D &rhs) const noexcept;
};

OPENACC_ROUTINE
Vector3D operator-(Vector3D v, Vector3D w);

OPENACC_ROUTINE
Vector3D operator*(Real scalar, Vector3D v);

OPENACC_ROUTINE
Vector3D operator*(Vector3D v, Real scalar);

OPENACC_ROUTINE
Vector3D operator/(Vector3D v, Real scalar);

struct ComplexVector3D
{
    Complex x, y, z;

    static inline ComplexVector3D from(const Vector3D &source)
    {
        return {source.x, source.y, source.z};
    }

    ComplexVector3D &operator+=(ComplexVector3D v) noexcept;
};

OPENACC_ROUTINE
ComplexVector3D operator+(ComplexVector3D v, ComplexVector3D w);

OPENACC_ROUTINE
ComplexVector3D operator-(ComplexVector3D v, ComplexVector3D w);

OPENACC_ROUTINE
ComplexVector3D operator*(Complex scalar, ComplexVector3D v);
