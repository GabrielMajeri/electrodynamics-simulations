from typing import TYPE_CHECKING, cast

import numpy as np

from electrodynamics.constants import SPEED_OF_LIGHT as c

if TYPE_CHECKING:
    import jax


def compute_particle_angular_momentum_z[ArrayT: (np.ndarray, "jax.Array")](
    positions: ArrayT, momenta: ArrayT
) -> ArrayT:
    """Computes the z-component of particle orbital angular momentum `L_z = x * p_y - y * p_x`.

    Accepts 4-vectors `(ct, x, y, z)` / `(p0, px, py, pz)` or 3-vectors `(x, y, z)` / `(px, py, pz)`.
    """
    if positions.shape[-1] not in (3, 4):
        raise ValueError(
            f"Positions must be a 3-vector or 4-vector, got shape {positions.shape}"
        )
    if momenta.shape[-1] not in (3, 4):
        raise ValueError(
            f"Momenta must be a 3-vector or 4-vector, got shape {momenta.shape}"
        )
    if positions.shape[-1] != momenta.shape[-1]:
        raise ValueError(
            f"Positions and momenta must have the same number of components, got {positions.shape[-1]} and {momenta.shape[-1]}"
        )

    if positions.shape[-1] == 4:
        x = positions[..., 1]
        y = positions[..., 2]
        px = momenta[..., 1]
        py = momenta[..., 2]
    else:
        x = positions[..., 0]
        y = positions[..., 1]
        px = momenta[..., 0]
        py = momenta[..., 1]

    return x * py - y * px


def compute_angular_momentum_derivative(
    detector_positions: np.ndarray, electric_field: np.ndarray, omega: float
) -> np.ndarray:
    """Computes the laboratory-frame z-component of radiated orbital angular momentum
    spectral density ``d^2 L_z / (d omega dA)`` in Gaussian atomic units, matching
    ``fields.py``'s Fourier transform convention
    ``E_tilde(x, omega) = (1 / 2*pi) * int E(x, t) * exp(+i * omega * t) dt``.

    Formula:
        ``d^2 L_z / (d omega dA) = (c / omega) * sum_i Re[(1 / i) * E_tilde_i^* * (x d/dy - y d/dx) E_tilde_i]``.

    For a wave with positive-frequency Fourier phase ``exp(+i * l * phi)``
    (corresponding to time-domain ``cos(omega * t - l * phi)``, lab-frame ``L_z = +l``),
    ``Re(dL_z) = +(l / omega) * c * |E_tilde|^2 = +(l / omega) * d^2 U / (d omega dA)``.
    Note that for a reflected beam propagating along ``-z_hat``, lab-frame ``L_z``
    retains its sign while beam-axis helicity is ``L_{k_hat} = -L_z``.
    """
    assert detector_positions.ndim == 3, (
        "Detector position should be a 2D array of 3D vectors"
    )
    assert detector_positions.shape[-1] == 3, (
        "Detector position should be an array of 3D vectors"
    )

    assert electric_field.ndim == 3, (
        "Electric field should be a 2D array of 3D complex phasors"
    )
    assert electric_field.shape[-1] == 3, (
        "Electric field should be an array of 3D complex phasors"
    )

    dx: float = (detector_positions[1, 0, 0] - detector_positions[0, 0, 0]).item()
    dy: float = (detector_positions[0, 1, 1] - detector_positions[0, 0, 1]).item()

    # Factor out the radially symmetric spherical propagation phase
    #   exp(i * (omega/c) * |r_det|)
    # before taking Cartesian finite differences. Since
    #   (x d/dy - y d/dx) |r_det| == 0
    # holds analytically, this is an exact identity that eliminates Cartesian grid dispersion
    # (which causes sin(4*phi) aliasing artifacts).
    r_det = np.linalg.norm(detector_positions, axis=-1, keepdims=True)
    r_center = float(
        r_det[detector_positions.shape[0] // 2, detector_positions.shape[1] // 2, 0]
    )
    spherical_phase = np.exp(-1j * (omega / c) * (r_det - r_center))
    smooth_field = electric_field * spherical_phase

    electric_field_dx, electric_field_dy = cast(
        tuple[np.ndarray, np.ndarray],
        np.gradient(smooth_field[:, :, :], dx, dy, axis=(0, 1)),
    )

    # Compute the z-component of the orbital angular momentum spectral density
    dL_z = np.zeros(detector_positions.shape[:2], dtype=np.complex128)

    for i in range(3):
        dL_z += (
            np.conjugate(smooth_field[:, :, i])
            / (1j * omega)
            * (
                detector_positions[:, :, 0] * electric_field_dy[:, :, i]
                - detector_positions[:, :, 1] * electric_field_dx[:, :, i]
            )
        )

    dL_z *= c

    return dL_z


def _integrate_detector_scalar_2d(
    detector_positions: np.ndarray, values: np.ndarray
) -> float:
    """Integrates a 2D scalar array over a rectangular detector grid using the trapezoidal rule."""
    if detector_positions.ndim != 3 or detector_positions.shape[-1] != 3:
        raise ValueError(
            f"detector_positions must have shape (Nx, Ny, 3), got {detector_positions.shape}"
        )
    if values.shape != detector_positions.shape[:2]:
        raise ValueError(
            f"values shape {values.shape} must match detector grid shape {detector_positions.shape[:2]}"
        )

    xs = detector_positions[:, 0, 0]
    ys = detector_positions[0, :, 1]
    area_sign = (1.0 if xs[-1] >= xs[0] else -1.0) * (1.0 if ys[-1] >= ys[0] else -1.0)
    return float(
        area_sign * np.trapezoid(np.trapezoid(values, xs, axis=0), ys, axis=0)
    )


def integrate_detector_oam_spectral_density(
    detector_positions: np.ndarray, dL_z: np.ndarray
) -> float:
    """Integrates the radiated OAM spectral density ``d^2 L_z / (d omega dA)``
    over a 2D rectangular detector grid to return ``dL_z / d omega``.
    """
    return _integrate_detector_scalar_2d(detector_positions, np.real(dL_z))


def compute_oam_per_photon(
    detector_positions: np.ndarray,
    electric_field: np.ndarray,
    dL_z: np.ndarray,
    omega: float,
    propagation_sign: int = 1,
) -> float:
    """Computes the dimensionless orbital angular momentum per photon
    ``<l> = propagation_sign * omega * (dL_z / d omega) / (dU / d omega)``
    from far-field detector fields.

    Set ``propagation_sign = 1`` for lab-frame ``L_z``, or ``-1`` for beam-axis helicity
    of a beam propagating along ``-z_hat``.
    """
    if propagation_sign not in (1, -1):
        raise ValueError(
            f"propagation_sign must be 1 or -1, got {propagation_sign}"
        )
    if electric_field.shape != (*detector_positions.shape[:2], 3):
        raise ValueError(
            f"electric_field shape {electric_field.shape} must match {(*detector_positions.shape[:2], 3)}"
        )

    total_dLz_domega = integrate_detector_oam_spectral_density(detector_positions, dL_z)
    dU_domega_dA = c * np.sum(np.abs(electric_field) ** 2, axis=-1)
    total_dU_domega = _integrate_detector_scalar_2d(detector_positions, dU_domega_dA)

    if total_dU_domega <= 0.0:
        return 0.0

    return float(propagation_sign * omega * total_dLz_domega / total_dU_domega)
