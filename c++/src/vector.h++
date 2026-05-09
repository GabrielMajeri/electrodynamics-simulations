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

    static Vector3D from_position(const Position &position)
    {
        return Vector3D(position.x, position.y, position.z);
    }
};

OPENACC_ROUTINE
Vector3D operator*(Real scalar, Vector3D v);
