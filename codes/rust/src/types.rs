use nalgebra::{Vector3, Vector4};

/// Type used for real (floating point) values.
pub type Real = f64;

/// 3-dimensional vector (e.g. position in Euclidean space, direction vector)
pub type Vec3 = Vector3<Real>;

/// 4-dimensional vector (e.g. 4-position, 4-momentum)
pub type Vec4 = Vector4<Real>;
