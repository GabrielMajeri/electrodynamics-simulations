#include "vector.h++"

Acceleration operator*(Real scalar, Acceleration acc)
{
    return Acceleration{scalar * acc.dgamma, scalar * acc.dvx, scalar * acc.dvy, scalar * acc.dvz};
}

Momentum operator+(Momentum m, Acceleration acc)
{
    return Momentum{m.gamma + acc.dgamma, m.vx + acc.dvx, m.vy + acc.dvy, m.vz + acc.dvz};
}

Momentum operator*(Real scalar, Momentum m)
{
    return Momentum{scalar * m.gamma, scalar * m.vx, scalar * m.vy, scalar * m.vz};
}

Position operator+(Position p, Momentum m)
{
    return Position{p.t + m.gamma, p.x + m.vx, p.y + m.vy, p.z + m.vz};
}

Vector3D operator*(Real scalar, Vector3D v)
{
    return Vector3D{scalar * v.x, scalar * v.y, scalar * v.z};
}
