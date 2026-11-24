from dataclasses import dataclass
from math import cos, exp, factorial, pi, sin, sqrt
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from scipy.special import eval_genlaguerre

from electrodynamics.beams import (
    GaussianBeamParameters,
    LaguerreGaussBeamParameters,
    LaserParameters,
)
from electrodynamics.constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_C,
    HARTREE_TO_EV,
    au_energy_to_ev,
    au_energy_to_joules,
)
from electrodynamics.constants import (
    SPEED_OF_LIGHT as c,
)
from electrodynamics.detector import (
    DiskDetectorParameters,
    compute_detector_area_weights,
    initialize_detector_positions_negative_z,
)
from electrodynamics.fields import (
    compute_acceleration_of_charged_particle_in_em_field,
    compute_relativistic_reflection_parameters,
)
from electrodynamics.integrate import compute_electric_and_magnetic_fields
from electrodynamics.pulse import (
    CosineSquaredPulseParameters,
    GaussianPulseParameters,
    PulseWithFlatPeakParameters,
)

type AnyLaserParameters = (
    LaserParameters | GaussianBeamParameters | LaguerreGaussBeamParameters
)


@dataclass(frozen=True)
class FrequencyQuadrature:
    "Gauss-Legendre frequency quadrature grid and integration weights."

    frequencies: np.ndarray
    weights: np.ndarray
    omega_min: float
    omega_max: float
    omega_center: float
    bandwidth_fraction: float


@dataclass(frozen=True)
class DetectorSpectralEnergyResult:
    "Single-frequency detector spectral energy dU/domega and far-field check."

    dU_domega: float
    dU_domega_far_field: float
    relative_far_field_mismatch: float
    spectral_energy_flux: np.ndarray


@dataclass(frozen=True)
class DetectorTotalEnergyResult:
    "Total frequency- and area-integrated radiation energy received on a detector."

    total_energy_au: float
    total_energy_joules: float
    total_energy_ev: float
    far_field_energy_au: float
    relative_far_field_mismatch: float
    fluence_per_pixel: np.ndarray
    spectrum_dU_domega: np.ndarray


@dataclass(frozen=True)
class FactorizedDetectorEnergyResult:
    "Fast factorized energy estimate (dU/domega)|_{omega_ref} * Delta_omega_eff."

    total_energy_au: float
    total_energy_joules: float
    total_energy_ev: float
    dU_domega_at_reference: float
    effective_bandwidth: float
    reference_frequency: float


@dataclass(frozen=True)
class ElectronKineticEnergyBalance:
    "Initial, final, and delta kinetic energy of the simulated electron ensemble."

    num_particles: int
    initial_au: float
    final_au: float
    delta_au: float
    initial_kev: float
    final_kev: float
    delta_kev: float
    initial_joules: float
    final_joules: float
    delta_joules: float
    mean_initial_gamma: float
    mean_final_gamma: float


@dataclass(frozen=True)
class ThinSheetReflectivityBenchmark:
    "Comparison against the analytical cold relativistic flying-mirror reflectivity model."

    surface_density_au: float
    coupling_alpha: float
    doppler_factor: float
    weak_photon_reflectivity: float
    saturated_photon_reflectivity: float
    weak_theory_energy_au: float
    saturated_theory_energy_au: float
    simulated_to_weak_ratio: float | None
    simulated_to_saturated_ratio: float | None


@dataclass(frozen=True)
class PhysicalBunchExtrapolation:
    "Coherent N_e^2 energy scaling and test-particle validity metrics for a real bunch charge."

    simulated_electrons: int
    physical_electrons: float
    bunch_charge_pc: float
    scale_factor: float
    detector_energy_au: float
    detector_energy_joules: float
    larmor_incoherent_energy_au: float | None
    photon_reflectivity: float
    electron_kinetic_energy_au: float | None
    radiated_to_kinetic_ratio: float | None
    electron_source_to_kinetic_ratio: float | None
    test_particle_regime_valid: bool | None


@dataclass(frozen=True)
class EnergyTransferAnalysis:
    "Complete test-particle energy accounting, transfer partition, and saturation bounds."

    detector_energy_au: float
    detector_energy_joules: float
    detector_energy_ev: float
    incident_energy_au: float
    incident_energy_joules: float
    intercepted_incident_energy_au: float | None
    interception_fraction: float | None
    laser_energy_fraction: float
    photon_reflectivity: float
    doppler_factor: float
    laser_energy_share_per_photon: float
    electron_energy_share_per_photon: float
    detector_energy_factorized_au: float | None
    factorized_to_exact_ratio: float | None
    larmor_energy_au: float | None
    coherence_gain: float | None
    electron_kinetic_energy: ElectronKineticEnergyBalance | None
    thin_sheet_benchmark: ThinSheetReflectivityBenchmark | None
    saturation_threshold: float
    saturation_electrons_reflectivity: float | None
    saturation_charge_pc_reflectivity: float | None
    saturation_electrons_kinetic: float | None
    saturation_charge_pc_kinetic: float | None
    physical_extrapolation: PhysicalBunchExtrapolation | None


def build_frequency_quadrature(
    omega_center: float,
    bandwidth_fraction: float = 0.6,
    num_frequencies: int = 16,
    *,
    omega_min: float | None = None,
    omega_max: float | None = None,
) -> FrequencyQuadrature:
    """Builds a Gauss-Legendre quadrature grid for integrating
    frequencies in the range `[omega_min, omega_max]`.

    When `omega_min` and `omega_max` are omitted, `bandwidth_fraction` specifies the
    full fractional width `Delta_omega / omega_center` centered at `omega_center`.
    """
    if num_frequencies < 1:
        raise ValueError(f"num_frequencies must be >= 1, got {num_frequencies}")

    if omega_min is not None or omega_max is not None:
        if omega_min is None or omega_max is None:
            raise ValueError("Provide both omega_min and omega_max, or neither.")
        if omega_min <= 0.0 or omega_max <= omega_min:
            raise ValueError(
                f"Require 0 < omega_min < omega_max, got ({omega_min}, {omega_max})"
            )
        resolved_min = float(omega_min)
        resolved_max = float(omega_max)
        resolved_center = 0.5 * (resolved_min + resolved_max)
        resolved_bw = (resolved_max - resolved_min) / resolved_center
    else:
        if omega_center <= 0.0:
            raise ValueError(f"omega_center must be positive, got {omega_center}")
        if not (0.0 < bandwidth_fraction < 2.0):
            raise ValueError(
                f"bandwidth_fraction must lie in (0, 2), got {bandwidth_fraction}"
            )
        half_width = 0.5 * bandwidth_fraction * omega_center
        resolved_center = float(omega_center)
        resolved_bw = float(bandwidth_fraction)
        resolved_min = resolved_center - half_width
        resolved_max = resolved_center + half_width

    xi, w_gl = np.polynomial.legendre.leggauss(num_frequencies)
    half_span = 0.5 * (resolved_max - resolved_min)
    frequencies = resolved_center + half_span * xi
    weights = half_span * w_gl

    return FrequencyQuadrature(
        frequencies=frequencies,
        weights=weights,
        omega_min=resolved_min,
        omega_max=resolved_max,
        omega_center=resolved_center,
        bandwidth_fraction=resolved_bw,
    )


def _normalize_normal_vector(
    normal_vector: tuple[float, float, float] | np.ndarray,
    expected_prefix_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    "Normalizes an array of surface-normal vectors to ensure they have unit length."

    normal = np.asarray(normal_vector, dtype=np.float64)
    if normal.ndim == 1:
        if normal.shape != (3,):
            raise ValueError(f"normal_vector must have shape (3,), got {normal.shape}")
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            raise ValueError("normal_vector must have nonzero norm")
        return normal / norm

    if expected_prefix_shape is not None and normal.shape != (
        *expected_prefix_shape,
        3,
    ):
        raise ValueError(
            f"Per-pixel normal_vector shape {normal.shape} must match {(*expected_prefix_shape, 3)}"
        )

    if normal.shape[-1] != 3:
        raise ValueError(
            f"normal_vector last axis must have length 3, got shape {normal.shape}"
        )

    norms = np.linalg.vector_norm(normal, axis=-1, keepdims=True)

    if np.any(norms <= 0.0):
        raise ValueError("All per-pixel normal vectors must have nonzero norm")

    return normal / norms


def compute_spectral_energy_flux(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    normal_vector: tuple[float, float, float] | np.ndarray = (0.0, 0.0, -1.0),
) -> np.ndarray:
    """Computes Parseval spectral energy flux density crossing `normal_vector`:
      d^2 U / (d omega dA) = c * Re[E_tilde x B_tilde^*] . n_hat
    where `E_tilde` and `B_tilde` are the complex Fourier amplitudes of the EM fields.
    """
    E = np.asarray(electric_field)
    B = np.asarray(magnetic_field)
    if E.shape != B.shape or E.shape[-1] != 3:
        raise ValueError(
            f"electric_field {E.shape} and magnetic_field {B.shape} must have identical shape (..., 3)"
        )

    normal = _normalize_normal_vector(normal_vector)
    if normal.ndim > 1 and normal.shape != E.shape:
        if normal.shape == (*E.shape[:-2], 3):
            normal = np.expand_dims(normal, axis=-2)
        else:
            raise ValueError(
                f"normal_vector shape {normal.shape} is incompatible with field shape {E.shape}"
            )

    poynting_spectral = c * np.real(np.cross(E, np.conjugate(B), axis=-1))
    return np.sum(poynting_spectral * normal, axis=-1)


def _compute_far_field_cos_theta(
    detector_positions: np.ndarray | None,
    normal_spatial: np.ndarray,
    spatial_shape: tuple[int, ...],
) -> np.ndarray | float:
    if detector_positions is None:
        return 1.0
    pos = np.asarray(detector_positions, dtype=np.float64)
    if pos.shape != (*spatial_shape, 3):
        raise ValueError(
            f"detector_positions shape {pos.shape} must match {(*spatial_shape, 3)}"
        )
    norms = np.linalg.vector_norm(pos, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    r_hat = np.where(norms > 0.0, pos / safe_norms, 0.0)
    return np.sum(r_hat * normal_spatial, axis=-1)


def compute_detector_spectral_energy(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    area_weights: np.ndarray,
    normal_vector: tuple[float, float, float] | np.ndarray = (0.0, 0.0, -1.0),
    detector_positions: np.ndarray | None = None,
) -> DetectorSpectralEnergyResult:
    """Integrates single-frequency spectral energy flux over detector area and checks
    consistency against the far-field approximation ``c |E|^2 (r_hat . n_hat)``.
    """
    E = np.asarray(electric_field)
    B = np.asarray(magnetic_field)
    weights = np.asarray(area_weights, dtype=np.float64)
    spatial_shape = weights.shape

    if E.shape != (*spatial_shape, 3) or B.shape != (*spatial_shape, 3):
        raise ValueError(
            f"Field shapes {E.shape}, {B.shape} must match {(*spatial_shape, 3)}"
        )

    normal_spatial = _normalize_normal_vector(normal_vector, spatial_shape)
    flux = compute_spectral_energy_flux(E, B, normal_vector=normal_spatial)
    dU_domega = float(np.sum(flux * weights))

    cos_theta = _compute_far_field_cos_theta(
        detector_positions, normal_spatial, spatial_shape
    )
    far_field_flux = c * np.sum(np.abs(E) ** 2, axis=-1) * cos_theta
    dU_domega_ff = float(np.sum(far_field_flux * weights))

    denom = max(abs(dU_domega), abs(dU_domega_ff))
    mismatch = abs(dU_domega - dU_domega_ff) / denom if denom > 0.0 else 0.0

    return DetectorSpectralEnergyResult(
        dU_domega=dU_domega,
        dU_domega_far_field=dU_domega_ff,
        relative_far_field_mismatch=float(mismatch),
        spectral_energy_flux=flux,
    )


def compute_detector_total_energy(
    electric_field_multifreq: np.ndarray,
    magnetic_field_multifreq: np.ndarray,
    area_weights: np.ndarray,
    frequency_weights: np.ndarray,
    normal_vector: tuple[float, float, float] | np.ndarray = (0.0, 0.0, -1.0),
    detector_positions: np.ndarray | None = None,
) -> DetectorTotalEnergyResult:
    """Integrates multi-frequency scattered EM fields over detector area and frequency."""
    E = np.asarray(electric_field_multifreq)
    B = np.asarray(magnetic_field_multifreq)
    area_w = np.asarray(area_weights, dtype=np.float64)
    freq_w = np.asarray(frequency_weights, dtype=np.float64)

    spatial_shape = area_w.shape
    num_freq = freq_w.shape[0]
    if freq_w.ndim != 1:
        raise ValueError(f"frequency_weights must be 1D, got shape {freq_w.shape}")
    expected_field_shape = (*spatial_shape, num_freq, 3)
    if E.shape != expected_field_shape or B.shape != expected_field_shape:
        raise ValueError(
            f"Multi-frequency field shapes {E.shape}, {B.shape} must match {expected_field_shape}"
        )

    normal_spatial = _normalize_normal_vector(normal_vector, spatial_shape)
    flux = compute_spectral_energy_flux(E, B, normal_vector=normal_spatial)

    fluence_per_pixel = np.sum(flux * freq_w, axis=-1)
    spatial_axes = tuple(range(area_w.ndim))
    area_broadcast = np.expand_dims(area_w, axis=-1)
    spectrum_dU_domega = np.sum(flux * area_broadcast, axis=spatial_axes)

    total_energy_au = float(np.sum(fluence_per_pixel * area_w))

    cos_theta = _compute_far_field_cos_theta(
        detector_positions, normal_spatial, spatial_shape
    )
    if isinstance(cos_theta, np.ndarray):
        cos_theta = cos_theta[..., None]
    far_field_flux = c * np.sum(np.abs(E) ** 2, axis=-1) * cos_theta
    far_field_energy_au = float(
        np.sum(np.sum(far_field_flux * freq_w, axis=-1) * area_w)
    )

    denom = max(abs(total_energy_au), abs(far_field_energy_au))
    mismatch = (
        abs(total_energy_au - far_field_energy_au) / denom if denom > 0.0 else 0.0
    )

    return DetectorTotalEnergyResult(
        total_energy_au=total_energy_au,
        total_energy_joules=au_energy_to_joules(total_energy_au),
        total_energy_ev=au_energy_to_ev(total_energy_au),
        far_field_energy_au=far_field_energy_au,
        relative_far_field_mismatch=float(mismatch),
        fluence_per_pixel=fluence_per_pixel,
        spectrum_dU_domega=spectrum_dU_domega,
    )


def compute_effective_bandwidth(
    frequencies: np.ndarray,
    spectrum_electric_field: np.ndarray,
    reference_frequency: float,
    frequency_weights: np.ndarray | None = None,
) -> float:
    """Computes effective spectral bandwidth ``Delta_omega_eff`` from a single-point spectrum:
    ``Delta_omega_eff = (int |E(omega)|^2 d omega) / |E(omega_ref)|^2``.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    E_spec = np.asarray(spectrum_electric_field)
    if freqs.ndim != 1 or freqs.size < 2:
        raise ValueError(
            f"frequencies must be a 1D array with >= 2 points, got {freqs.shape}"
        )
    if E_spec.shape != (freqs.size, 3):
        raise ValueError(
            f"spectrum_electric_field must have shape {(freqs.size, 3)}, got {E_spec.shape}"
        )

    order = np.argsort(freqs)
    freqs_sorted = freqs[order]
    intensity_sorted = np.sum(np.abs(E_spec[order]) ** 2, axis=-1)

    if not (freqs_sorted[0] <= reference_frequency <= freqs_sorted[-1]):
        raise ValueError(
            f"reference_frequency={reference_frequency} lies outside spectrum range "
            f"[{freqs_sorted[0]}, {freqs_sorted[-1]}]"
        )

    if frequency_weights is not None:
        weights_arr = np.asarray(frequency_weights, dtype=np.float64)
        if weights_arr.shape != freqs.shape:
            raise ValueError(
                f"frequency_weights shape {weights_arr.shape} must match {freqs.shape}"
            )
        numerator = float(np.sum(intensity_sorted * weights_arr[order]))
    else:
        numerator = float(np.trapezoid(intensity_sorted, freqs_sorted))

    ref_intensity = float(
        np.interp(reference_frequency, freqs_sorted, intensity_sorted)
    )
    if ref_intensity <= 0.0:
        return 0.0
    return float(numerator / ref_intensity)


def compute_factorized_detector_energy(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    area_weights: np.ndarray,
    spectrum_frequencies: np.ndarray,
    spectrum_electric_field: np.ndarray,
    reference_frequency: float,
    normal_vector: tuple[float, float, float] | np.ndarray = (0.0, 0.0, -1.0),
    spectrum_frequency_weights: np.ndarray | None = None,
) -> FactorizedDetectorEnergyResult:
    """Estimates total detector energy via ``(dU_det / d omega)|_{omega_ref} * Delta_omega_eff``."""
    spectral = compute_detector_spectral_energy(
        electric_field,
        magnetic_field,
        area_weights,
        normal_vector=normal_vector,
    )
    delta_omega_eff = compute_effective_bandwidth(
        spectrum_frequencies,
        spectrum_electric_field,
        reference_frequency,
        frequency_weights=spectrum_frequency_weights,
    )
    total_energy_au = spectral.dU_domega * delta_omega_eff
    return FactorizedDetectorEnergyResult(
        total_energy_au=float(total_energy_au),
        total_energy_joules=au_energy_to_joules(total_energy_au),
        total_energy_ev=au_energy_to_ev(total_energy_au),
        dU_domega_at_reference=spectral.dU_domega,
        effective_bandwidth=delta_omega_eff,
        reference_frequency=float(reference_frequency),
    )


def compute_pulse_energy_integral(pulse_parameters: GaussianPulseParameters) -> float:
    """Computes the exact temporal envelope energy integral ``T_g = int_{-inf}^{inf} g(phi)^2 d phi``."""
    if isinstance(pulse_parameters, CosineSquaredPulseParameters):
        return 0.75 * pulse_parameters.tau_0
    if isinstance(pulse_parameters, PulseWithFlatPeakParameters):
        return (
            pulse_parameters.tau_0 * sqrt(pi / 2.0)
            + pulse_parameters.peak_duration_periods * pulse_parameters.tau_0
        )
    if isinstance(pulse_parameters, GaussianPulseParameters):
        return pulse_parameters.tau_0 * sqrt(pi / 2.0)
    raise TypeError(f"Unsupported pulse parameters type: {type(pulse_parameters)}")


def compute_incident_beam_fluence(
    laser_parameters: AnyLaserParameters,
    pulse_parameters: GaussianPulseParameters,
) -> float:
    """Computes peak cycle-averaged laser fluence ``F_inc = (c / 8*pi) * E_0^2 * T_g``."""
    T_g = compute_pulse_energy_integral(pulse_parameters)
    return float((c / (8.0 * pi)) * (laser_parameters.amplitude**2) * T_g)


def _eval_transverse_mode_intensity_normalized(
    laser_parameters: GaussianBeamParameters | LaguerreGaussBeamParameters,
    x_b: np.ndarray,
    y_b: np.ndarray,
    z_b: np.ndarray,
) -> np.ndarray:
    """Evaluates ``|u(x_b, y_b, z_b)|^2 / E_0^2`` in the beam's propagation frame."""
    w_0 = laser_parameters.waist_radius
    z_R = (pi * w_0**2) / laser_parameters.wavelength
    w_z = w_0 * np.sqrt(1.0 + (z_b / z_R) ** 2)
    r_sq = x_b**2 + y_b**2
    r_over_w_sq = r_sq / (w_z**2)

    if isinstance(laser_parameters, LaguerreGaussBeamParameters):
        p = laser_parameters.radial_index
        abs_l = abs(laser_parameters.azimuthal_index)
        norm_sq = 2.0 * factorial(p) / factorial(p + abs_l)
        xi = 2.0 * r_over_w_sq
        L_val = eval_genlaguerre(p, abs_l, xi)
        return norm_sq * ((w_0 / w_z) ** 2) * (xi**abs_l) * np.exp(-xi) * (L_val**2)

    return ((w_0 / w_z) ** 2) * np.exp(-2.0 * r_over_w_sq)


def compute_incident_beam_interception_fraction(
    laser_parameters: GaussianBeamParameters | LaguerreGaussBeamParameters,
    aperture_radius: float,
    z_plane: float = 0.0,
    incidence_angle: float = 0.0,
    num_radial: int = 48,
    num_azimuthal: int = 32,
) -> float:
    """Computes the fraction of incident Gaussian or LG beam energy passing through
    a circular aperture of radius ``aperture_radius`` on ``z = z_plane``.
    """
    if aperture_radius < 0.0:
        raise ValueError(f"aperture_radius must be non-negative, got {aperture_radius}")
    if aperture_radius == 0.0:
        return 0.0

    w_0 = laser_parameters.waist_radius
    z_R = (pi * w_0**2) / laser_parameters.wavelength
    w_z = w_0 * sqrt(1.0 + (z_plane / z_R) ** 2)

    if abs(incidence_angle) < 1e-12:
        xi_max = 2.0 * (aperture_radius / w_z) ** 2
        if isinstance(laser_parameters, LaguerreGaussBeamParameters):
            p = laser_parameters.radial_index
            abs_l = abs(laser_parameters.azimuthal_index)
            xi_upper = min(xi_max, 80.0 + 4.0 * p + 2.0 * abs_l)
            n_nodes = max(num_radial, 2 * p + abs_l + 32)
            t_gl, w_gl = np.polynomial.legendre.leggauss(n_nodes)
            xi = 0.5 * xi_upper * (t_gl + 1.0)
            w_xi = 0.5 * xi_upper * w_gl
            L_val = eval_genlaguerre(p, abs_l, xi)
            integrand = (xi**abs_l) * np.exp(-xi) * (L_val**2)
            frac = (factorial(p) / factorial(p + abs_l)) * float(
                np.sum(integrand * w_xi)
            )
            return float(np.clip(frac, 0.0, 1.0))

        return float(np.clip(1.0 - exp(-xi_max), 0.0, 1.0))

    disk_params = DiskDetectorParameters(
        radius=aperture_radius,
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        radial_sampling="gauss-legendre",
    )
    area_weights = compute_detector_area_weights(disk_params).reshape(-1)
    pos_flat = initialize_detector_positions_negative_z(disk_params, z_plane)
    x = pos_flat[:, 0]
    y = pos_flat[:, 1]

    cos_t = cos(incidence_angle)
    sin_t = sin(incidence_angle)
    x_b = x * cos_t - z_plane * sin_t
    y_b = y
    z_b = x * sin_t + z_plane * cos_t

    u_sq_norm = _eval_transverse_mode_intensity_normalized(
        laser_parameters, x_b, y_b, z_b
    )
    total_transverse_norm = (
        pi * w_0**2
        if isinstance(laser_parameters, LaguerreGaussBeamParameters)
        else 0.5 * pi * w_0**2
    )
    frac = float(np.sum(cos_t * u_sq_norm * area_weights) / total_transverse_norm)
    return float(np.clip(frac, 0.0, 1.0))


def compute_incident_beam_energy(
    laser_parameters: AnyLaserParameters,
    pulse_parameters: GaussianPulseParameters,
    aperture_radius: float | None = None,
    z_plane: float = 0.0,
    incidence_angle: float = 0.0,
) -> float:
    """Analytically computes total incident laser pulse energy (in Hartree a.u.),
    accounting for the intentional normalization difference between
    ``LaguerreGaussBeamParameters`` and ``GaussianBeamParameters``.

    When ``aperture_radius`` is provided, returns the energy intercepted by a disk of
    that radius on ``z = z_plane`` (required for plane-wave ``LaserParameters``).
    """
    T_g = compute_pulse_energy_integral(pulse_parameters)
    E_0 = laser_parameters.amplitude

    if isinstance(laser_parameters, LaguerreGaussBeamParameters):
        u_total = (c / 8.0) * (laser_parameters.waist_radius**2) * (E_0**2) * T_g
        if aperture_radius is None:
            return float(u_total)
        frac = compute_incident_beam_interception_fraction(
            laser_parameters,
            aperture_radius,
            z_plane=z_plane,
            incidence_angle=incidence_angle,
        )
        return float(u_total * frac)

    if isinstance(laser_parameters, GaussianBeamParameters):
        u_total = (c / 16.0) * (laser_parameters.waist_radius**2) * (E_0**2) * T_g
        if aperture_radius is None:
            return float(u_total)
        frac = compute_incident_beam_interception_fraction(
            laser_parameters,
            aperture_radius,
            z_plane=z_plane,
            incidence_angle=incidence_angle,
        )
        return float(u_total * frac)

    if aperture_radius is None:
        raise ValueError(
            "aperture_radius is required to compute finite energy for a plane-wave LaserParameters."
        )
    if aperture_radius < 0.0:
        raise ValueError(f"aperture_radius must be non-negative, got {aperture_radius}")
    fluence = compute_incident_beam_fluence(laser_parameters, pulse_parameters)
    return float(fluence * pi * (aperture_radius**2) * cos(incidence_angle))


def compute_incident_beam_energy_numerical(
    laser_parameters: AnyLaserParameters,
    pulse_parameters: GaussianPulseParameters,
    *,
    aperture_radius: float | None = None,
    z_plane: float = 0.0,
    incidence_angle: float = 0.0,
    num_radial: int = 48,
    num_azimuthal: int | None = None,
    points_per_period: int = 24,
) -> float:
    """Numerically integrates the 3D Gaussian Poynting flux ``(c / 4*pi) * (E x B_Gauss)_z``
    over the disk ``x^2 + y^2 <= R^2`` at ``z = z_plane`` and time ``t``.

    Converts ``beams.py``'s ``B_beam = B_Gauss / c`` back to Gaussian units via
    ``S_z = (c^2 / (4*pi)) * (E_x B_beam_y - E_y B_beam_x)``.
    """
    cos_t = cos(incidence_angle)
    sin_t = sin(incidence_angle)

    if aperture_radius is not None:
        if aperture_radius <= 0.0:
            raise ValueError(f"aperture_radius must be positive, got {aperture_radius}")
        r_max = float(aperture_radius)
    else:
        if not isinstance(laser_parameters, GaussianBeamParameters):
            raise ValueError(
                "aperture_radius is required for plane-wave LaserParameters."
            )
        w_0 = laser_parameters.waist_radius
        z_R = (pi * w_0**2) / laser_parameters.wavelength
        w_z = w_0 * sqrt(1.0 + (z_plane / z_R) ** 2)
        if isinstance(laser_parameters, LaguerreGaussBeamParameters):
            mode_factor = sqrt(
                (
                    2 * laser_parameters.radial_index
                    + abs(laser_parameters.azimuthal_index)
                    + 1
                )
                / 2.0
            )
        else:
            mode_factor = sqrt(0.5)
        r_max = (mode_factor + 5.0) * w_z / max(abs(cos_t), 0.1)

    abs_l = (
        abs(laser_parameters.azimuthal_index)
        if isinstance(laser_parameters, LaguerreGaussBeamParameters)
        else 0
    )
    n_phi = num_azimuthal if num_azimuthal is not None else max(32, 4 * abs_l + 16)

    disk_params = DiskDetectorParameters(
        radius=r_max,
        num_radial=num_radial,
        num_azimuthal=n_phi,
        radial_sampling="gauss-legendre",
    )
    area_weights_flat = jnp.asarray(
        compute_detector_area_weights(disk_params).reshape(-1)
    )
    spatial_pos = jnp.asarray(
        initialize_detector_positions_negative_z(disk_params, z_plane)
    )

    tau_0 = pulse_parameters.tau_0
    phi_0 = pulse_parameters.phi_0
    if isinstance(pulse_parameters, CosineSquaredPulseParameters):
        phi_lo = phi_0 - tau_0
        phi_hi = phi_0 + tau_0
    elif isinstance(pulse_parameters, PulseWithFlatPeakParameters):
        phi_lo = phi_0 - 5.0 * tau_0
        phi_hi = phi_0 + (pulse_parameters.peak_duration_periods + 5.0) * tau_0
    else:
        phi_lo = phi_0 - 5.0 * tau_0
        phi_hi = phi_0 + 5.0 * tau_0

    z_sweep = (z_plane * cos_t) / c
    x_sweep = (r_max * abs(sin_t)) / c
    t_min = phi_lo + z_sweep - x_sweep
    t_max = phi_hi + z_sweep + x_sweep

    laser_period = (2.0 * pi) / laser_parameters.frequency
    dt_target = min(laser_period / points_per_period, tau_0 / 32.0)
    num_time = max(16, int(np.ceil((t_max - t_min) / dt_target)) + 1)
    times = np.linspace(t_min, t_max, num_time, dtype=np.float64)
    dt = times[1] - times[0]
    time_weights = np.full(num_time, dt, dtype=np.float64)
    time_weights[0] *= 0.5
    time_weights[-1] *= 0.5

    @jdc.jit
    def integrate_time_steps(ct_arr: jax.Array, wt_arr: jax.Array) -> jax.Array:
        def step_fn(
            acc: jax.Array, item: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, None]:
            ct_val, wt_val = item
            ct_col = jnp.full((spatial_pos.shape[0], 1), ct_val)
            pos4 = jnp.concatenate([ct_col, spatial_pos], axis=-1)
            E, B_beam = compute_electric_and_magnetic_fields(
                pos4,
                laser_parameters,
                pulse_parameters,
                incidence_angle=incidence_angle,
            )
            S_z = ((c**2) / (4.0 * pi)) * (
                E[:, 0] * B_beam[:, 1] - E[:, 1] * B_beam[:, 0]
            )
            return acc + wt_val * jnp.sum(S_z * area_weights_flat), None

        total, _ = jax.lax.scan(
            step_fn, jnp.array(0.0, dtype=jnp.float64), (ct_arr, wt_arr)
        )
        return total

    total_energy = integrate_time_steps(
        jnp.asarray(times * c), jnp.asarray(time_weights)
    )
    return float(total_energy)


def compute_electron_kinetic_energy(
    momenta: np.ndarray | jax.Array,
    particle_mass: float = ELECTRON_MASS,
) -> float:
    """Computes total relativistic kinetic energy ``sum_i (gamma_i - 1) m c^2`` in Hartree a.u."""
    mom = np.asarray(momenta, dtype=np.float64)
    if mom.ndim != 2 or mom.shape[-1] != 4:
        raise ValueError(f"momenta must have shape (N, 4), got {mom.shape}")
    return float(np.sum(c * (mom[:, 0] - particle_mass * c)))


def compute_electron_kinetic_energy_balance(
    initial_momenta: np.ndarray | jax.Array,
    final_momenta: np.ndarray | jax.Array,
    particle_mass: float = ELECTRON_MASS,
) -> ElectronKineticEnergyBalance:
    """Computes initial, final, and change in ensemble electron kinetic energy."""
    init_mom = np.asarray(initial_momenta, dtype=np.float64)
    final_mom = np.asarray(final_momenta, dtype=np.float64)
    if (
        init_mom.shape != final_mom.shape
        or init_mom.ndim != 2
        or init_mom.shape[-1] != 4
    ):
        raise ValueError(
            f"initial_momenta {init_mom.shape} and final_momenta {final_mom.shape} "
            "must share shape (N, 4)"
        )

    num_particles = int(init_mom.shape[0])
    initial_au = compute_electron_kinetic_energy(init_mom, particle_mass=particle_mass)
    final_au = compute_electron_kinetic_energy(final_mom, particle_mass=particle_mass)
    delta_au = final_au - initial_au

    mc = particle_mass * c
    mean_init_gamma = float(np.mean(init_mom[:, 0] / mc)) if num_particles > 0 else 1.0
    mean_final_gamma = (
        float(np.mean(final_mom[:, 0] / mc)) if num_particles > 0 else 1.0
    )

    return ElectronKineticEnergyBalance(
        num_particles=num_particles,
        initial_au=initial_au,
        final_au=final_au,
        delta_au=delta_au,
        initial_kev=(initial_au * HARTREE_TO_EV) / 1000.0,
        final_kev=(final_au * HARTREE_TO_EV) / 1000.0,
        delta_kev=(delta_au * HARTREE_TO_EV) / 1000.0,
        initial_joules=au_energy_to_joules(initial_au),
        final_joules=au_energy_to_joules(final_au),
        delta_joules=au_energy_to_joules(delta_au),
        mean_initial_gamma=mean_init_gamma,
        mean_final_gamma=mean_final_gamma,
    )


@jdc.jit
def compute_larmor_power_from_derivatives(
    momentum: jax.Array,
    d_momentum: jax.Array,
    is_lab_time: jdc.Static[bool] = True,
    particle_charge: jdc.Static[float] = ELECTRON_CHARGE,
    particle_mass: jdc.Static[float] = ELECTRON_MASS,
) -> jax.Array:
    """Computes instantaneous relativistic Larmor radiated power ``dU_Larmor / dt``.

    When ``is_lab_time=True``, ``d_momentum`` is ``du^mu / dt`` and the invariant
    squared 4-acceleration is scaled by ``gamma^2 = (u^0 / (m c))^2``. When ``False``,
    ``d_momentum`` is proper-time acceleration ``du^mu / dtau``.
    """
    du0 = d_momentum[..., 0]
    du_spatial_sq = jnp.sum(jnp.square(d_momentum[..., 1:4]), axis=-1)
    invariant_accel_sq = jnp.maximum(du_spatial_sq - jnp.square(du0), 0.0)

    if is_lab_time:
        gamma_sq = jnp.square(momentum[..., 0] / (particle_mass * c))
        invariant_accel_sq = gamma_sq * invariant_accel_sq

    prefactor = (2.0 * (particle_charge**2)) / (3.0 * (particle_mass**2) * (c**3))
    return prefactor * invariant_accel_sq


@jdc.jit
def compute_larmor_radiated_energy_step(
    position: jax.Array,
    momentum: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[AnyLaserParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    incidence_angle: jdc.Static[float] = 0.0,
    is_lab_time: jdc.Static[bool] = True,
    particle_charge: jdc.Static[float] = ELECTRON_CHARGE,
    particle_mass: jdc.Static[float] = ELECTRON_MASS,
) -> jax.Array:
    """Computes per-particle Larmor radiated energy increment ``P_Larmor * dt`` over one step."""
    electric_field, magnetic_field = compute_electric_and_magnetic_fields(
        position, laser_parameters, pulse_parameters, incidence_angle=incidence_angle
    )
    du_dtau = compute_acceleration_of_charged_particle_in_em_field(
        momentum,
        electric_field,
        magnetic_field,
        charge_to_mass_ratio=particle_charge / particle_mass,
    )
    power = compute_larmor_power_from_derivatives(
        momentum,
        du_dtau,
        is_lab_time=False,
        particle_charge=particle_charge,
        particle_mass=particle_mass,
    )
    if is_lab_time:
        return power * time_step

    gamma = momentum[..., 0] / (particle_mass * c)
    return power * gamma * time_step


@jdc.jit
def accumulate_larmor_radiated_energy(
    positions_history: jax.Array,
    momenta_history: jax.Array,
    time_step: float,
    laser_parameters: jdc.Static[AnyLaserParameters],
    pulse_parameters: jdc.Static[GaussianPulseParameters],
    incidence_angle: jdc.Static[float] = 0.0,
    is_lab_time: jdc.Static[bool] = True,
    particle_charge: jdc.Static[float] = ELECTRON_CHARGE,
    particle_mass: jdc.Static[float] = ELECTRON_MASS,
) -> jax.Array:
    """Accumulates per-particle Larmor radiated energy over a trajectory history
    of shape ``(num_steps, num_particles, 4)`` using composite trapezoidal integration.
    """
    num_steps = positions_history.shape[0]
    num_particles = positions_history.shape[1]
    weights = jnp.ones((num_steps,), dtype=jnp.float64)
    if num_steps >= 2:
        weights = weights.at[0].set(0.5).at[-1].set(0.5)

    def scan_fn(
        acc: jax.Array, item: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, None]:
        pos, mom, w = item
        dU = compute_larmor_radiated_energy_step(
            pos,
            mom,
            time_step,
            laser_parameters,
            pulse_parameters,
            incidence_angle=incidence_angle,
            is_lab_time=is_lab_time,
            particle_charge=particle_charge,
            particle_mass=particle_mass,
        )
        return acc + w * dU, None

    total_per_particle, _ = jax.lax.scan(
        scan_fn,
        jnp.zeros((num_particles,), dtype=jnp.float64),
        (positions_history, momenta_history, weights),
    )
    return total_per_particle


def compute_thin_sheet_reflectivity(
    num_electrons: float,
    sheet_radius: float,
    gamma: float,
    laser_frequency: float,
    incidence_angle: float = 0.0,
    incident_energy_au: float = 0.0,
    doppler_factor: float | None = None,
    simulated_detector_energy_au: float = 0.0,
    particle_charge: float = ELECTRON_CHARGE,
    particle_mass: float = ELECTRON_MASS,
) -> ThinSheetReflectivityBenchmark:
    """Computes the cold thin-sheet relativistic flying-mirror reflectivity benchmark:
    ``alpha = 2*pi*q^2*sigma_e / (m_e * c * gamma * (1 + beta*cos(theta_in)) * omega_0)``,
    ``R_weak = alpha^2``, ``R_sat = alpha^2 / (1 + alpha^2)``, ``U_rad = D * R * U_inc``.
    """
    if sheet_radius <= 0.0:
        raise ValueError(f"sheet_radius must be positive, got {sheet_radius}")
    if gamma < 1.0:
        raise ValueError(f"gamma must be >= 1, got {gamma}")
    if laser_frequency <= 0.0:
        raise ValueError(f"laser_frequency must be positive, got {laser_frequency}")

    sigma_e = float(num_electrons) / (pi * (sheet_radius**2))
    beta = sqrt(1.0 - 1.0 / (gamma**2))
    alpha = (2.0 * pi * (particle_charge**2) * sigma_e) / (
        particle_mass
        * c
        * gamma
        * (1.0 + beta * cos(incidence_angle))
        * laser_frequency
    )

    resolved_doppler = (
        float(doppler_factor)
        if doppler_factor is not None
        else compute_relativistic_reflection_parameters(gamma, incidence_angle)[2]
    )

    r_weak = alpha**2
    r_sat = r_weak / (1.0 + r_weak)
    u_weak = resolved_doppler * r_weak * incident_energy_au
    u_sat = resolved_doppler * r_sat * incident_energy_au

    sim_to_weak = (
        simulated_detector_energy_au / u_weak
        if (simulated_detector_energy_au > 0.0 and u_weak > 0.0)
        else None
    )
    sim_to_sat = (
        simulated_detector_energy_au / u_sat
        if (simulated_detector_energy_au > 0.0 and u_sat > 0.0)
        else None
    )

    return ThinSheetReflectivityBenchmark(
        surface_density_au=sigma_e,
        coupling_alpha=alpha,
        doppler_factor=resolved_doppler,
        weak_photon_reflectivity=r_weak,
        saturated_photon_reflectivity=r_sat,
        weak_theory_energy_au=u_weak,
        saturated_theory_energy_au=u_sat,
        simulated_to_weak_ratio=sim_to_weak,
        simulated_to_saturated_ratio=sim_to_sat,
    )


def analyze_energy_transfer(
    detector_energy_au: float,
    incident_energy_au: float,
    doppler_factor: float,
    num_electrons: int,
    *,
    detector_energy_factorized_au: float | None = None,
    intercepted_incident_energy_au: float | None = None,
    larmor_energy_au: float | None = None,
    electron_kinetic_energy: ElectronKineticEnergyBalance | None = None,
    sheet_radius: float | None = None,
    gamma: float | None = None,
    laser_frequency: float | None = None,
    incidence_angle: float = 0.0,
    physical_num_electrons: float | None = None,
    physical_bunch_charge_pc: float | None = None,
    saturation_threshold: float = 0.1,
) -> EnergyTransferAnalysis:
    """Synthesizes detector energy, incident laser energy, kinetic energy balance,
    Larmor emission, thin-sheet reflectivity theory, and real bunch-charge scaling.
    """
    if incident_energy_au <= 0.0:
        raise ValueError(
            f"incident_energy_au must be positive, got {incident_energy_au}"
        )
    if doppler_factor <= 0.0:
        raise ValueError(f"doppler_factor must be positive, got {doppler_factor}")
    if num_electrons < 1:
        raise ValueError(f"num_electrons must be >= 1, got {num_electrons}")
    if physical_num_electrons is not None and physical_bunch_charge_pc is not None:
        raise ValueError(
            "Specify at most one of physical_num_electrons or physical_bunch_charge_pc."
        )

    eta_laser = detector_energy_au / incident_energy_au
    photon_reflectivity = eta_laser / doppler_factor
    laser_share = 1.0 / doppler_factor
    electron_share = (doppler_factor - 1.0) / doppler_factor

    interception_fraction = (
        intercepted_incident_energy_au / incident_energy_au
        if intercepted_incident_energy_au is not None
        else None
    )
    factorized_ratio = (
        detector_energy_factorized_au / detector_energy_au
        if (detector_energy_factorized_au is not None and detector_energy_au > 0.0)
        else None
    )
    coherence_gain = (
        detector_energy_au / larmor_energy_au
        if (larmor_energy_au is not None and larmor_energy_au > 0.0)
        else None
    )

    thin_sheet_benchmark = None
    if sheet_radius is not None and gamma is not None and laser_frequency is not None:
        thin_sheet_benchmark = compute_thin_sheet_reflectivity(
            num_electrons=num_electrons,
            sheet_radius=sheet_radius,
            gamma=gamma,
            laser_frequency=laser_frequency,
            incidence_angle=incidence_angle,
            incident_energy_au=incident_energy_au,
            doppler_factor=doppler_factor,
            simulated_detector_energy_au=detector_energy_au,
        )

    pc_per_electron = ELEMENTARY_CHARGE_C * 1e12
    sat_n_refl = (
        num_electrons * sqrt(saturation_threshold / photon_reflectivity)
        if photon_reflectivity > 0.0
        else None
    )
    sat_q_refl = sat_n_refl * pc_per_electron if sat_n_refl is not None else None

    sat_n_kin = None
    sat_q_kin = None
    if (
        electron_kinetic_energy is not None
        and electron_kinetic_energy.initial_au > 0.0
        and detector_energy_au > 0.0
    ):
        sat_n_kin = (
            num_electrons
            * saturation_threshold
            * electron_kinetic_energy.initial_au
            / detector_energy_au
        )
        sat_q_kin = sat_n_kin * pc_per_electron

    physical_extrapolation = None
    if physical_num_electrons is not None or physical_bunch_charge_pc is not None:
        if physical_num_electrons is not None:
            n_phys = float(physical_num_electrons)
            q_phys_pc = n_phys * pc_per_electron
        else:
            assert physical_bunch_charge_pc is not None
            q_phys_pc = float(physical_bunch_charge_pc)
            n_phys = q_phys_pc / pc_per_electron

        scale = n_phys / num_electrons
        u_det_phys = (scale**2) * detector_energy_au
        u_larmor_phys = (
            scale * larmor_energy_au if larmor_energy_au is not None else None
        )
        r_photon_phys = (scale**2) * photon_reflectivity

        k_e_phys = None
        rad_to_kin = None
        src_to_kin = None
        regime_valid = None
        if electron_kinetic_energy is not None:
            k_e_phys = scale * electron_kinetic_energy.initial_au
            if k_e_phys > 0.0:
                rad_to_kin = u_det_phys / k_e_phys
                src_to_kin = (electron_share * u_det_phys) / k_e_phys
                regime_valid = (
                    r_photon_phys < saturation_threshold
                    and rad_to_kin < saturation_threshold
                )

        physical_extrapolation = PhysicalBunchExtrapolation(
            simulated_electrons=num_electrons,
            physical_electrons=n_phys,
            bunch_charge_pc=q_phys_pc,
            scale_factor=scale,
            detector_energy_au=u_det_phys,
            detector_energy_joules=au_energy_to_joules(u_det_phys),
            larmor_incoherent_energy_au=u_larmor_phys,
            photon_reflectivity=r_photon_phys,
            electron_kinetic_energy_au=k_e_phys,
            radiated_to_kinetic_ratio=rad_to_kin,
            electron_source_to_kinetic_ratio=src_to_kin,
            test_particle_regime_valid=regime_valid,
        )

    return EnergyTransferAnalysis(
        detector_energy_au=float(detector_energy_au),
        detector_energy_joules=au_energy_to_joules(detector_energy_au),
        detector_energy_ev=au_energy_to_ev(detector_energy_au),
        incident_energy_au=float(incident_energy_au),
        incident_energy_joules=au_energy_to_joules(incident_energy_au),
        intercepted_incident_energy_au=(
            float(intercepted_incident_energy_au)
            if intercepted_incident_energy_au is not None
            else None
        ),
        interception_fraction=interception_fraction,
        laser_energy_fraction=float(eta_laser),
        photon_reflectivity=float(photon_reflectivity),
        doppler_factor=float(doppler_factor),
        laser_energy_share_per_photon=float(laser_share),
        electron_energy_share_per_photon=float(electron_share),
        detector_energy_factorized_au=(
            float(detector_energy_factorized_au)
            if detector_energy_factorized_au is not None
            else None
        ),
        factorized_to_exact_ratio=factorized_ratio,
        larmor_energy_au=(
            float(larmor_energy_au) if larmor_energy_au is not None else None
        ),
        coherence_gain=coherence_gain,
        electron_kinetic_energy=electron_kinetic_energy,
        thin_sheet_benchmark=thin_sheet_benchmark,
        saturation_threshold=float(saturation_threshold),
        saturation_electrons_reflectivity=sat_n_refl,
        saturation_charge_pc_reflectivity=sat_q_refl,
        saturation_electrons_kinetic=sat_n_kin,
        saturation_charge_pc_kinetic=sat_q_kin,
        physical_extrapolation=physical_extrapolation,
    )


def build_energy_summary_dict(
    analysis: EnergyTransferAnalysis,
    *,
    detector_spectral_energy: DetectorSpectralEnergyResult | None = None,
    detector_total_energy: DetectorTotalEnergyResult | None = None,
    factorized_detector_energy: FactorizedDetectorEnergyResult | None = None,
    frequency_quadrature: FrequencyQuadrature | None = None,
    energy_method: str = "multifreq-quadrature",
) -> dict[str, Any]:
    """Builds a compact JSON-serializable energy summary dictionary for `save_json`."""
    spectral_summary = (
        {
            "dU_domega": detector_spectral_energy.dU_domega,
            "dU_domega_far_field": detector_spectral_energy.dU_domega_far_field,
            "relative_far_field_mismatch": (
                detector_spectral_energy.relative_far_field_mismatch
            ),
        }
        if detector_spectral_energy is not None
        else None
    )
    total_summary = (
        {
            "total_energy_au": detector_total_energy.total_energy_au,
            "total_energy_joules": detector_total_energy.total_energy_joules,
            "total_energy_ev": detector_total_energy.total_energy_ev,
            "far_field_energy_au": detector_total_energy.far_field_energy_au,
            "relative_far_field_mismatch": (
                detector_total_energy.relative_far_field_mismatch
            ),
        }
        if detector_total_energy is not None
        else None
    )
    quad_summary = (
        {
            "num_frequencies": int(len(frequency_quadrature.frequencies)),
            "omega_min": frequency_quadrature.omega_min,
            "omega_max": frequency_quadrature.omega_max,
            "omega_center": frequency_quadrature.omega_center,
            "bandwidth_fraction": frequency_quadrature.bandwidth_fraction,
        }
        if frequency_quadrature is not None
        else None
    )

    return {
        "energy_method": energy_method,
        "detector_spectral_energy": spectral_summary,
        "detector_total_energy": total_summary,
        "factorized_detector_energy": factorized_detector_energy,
        "frequency_quadrature": quad_summary,
        "energy_transfer_analysis": analysis,
    }


def format_energy_transfer_report(
    analysis: EnergyTransferAnalysis,
    *,
    detector_spectral_energy: DetectorSpectralEnergyResult | None = None,
    detector_total_energy: DetectorTotalEnergyResult | None = None,
    factorized_detector_energy: FactorizedDetectorEnergyResult | None = None,
) -> str:
    """Formats a human-readable multi-line energy accounting summary."""
    lines = ["=== Energy Accounting Summary ==="]

    if detector_total_energy is not None:
        lines.append(
            f"Detector energy (exact multi-freq): {analysis.detector_energy_au:.4e} Ha "
            f"({analysis.detector_energy_ev:.4e} eV, {analysis.detector_energy_joules:.4e} J) "
            f"| far-field mismatch = {detector_total_energy.relative_far_field_mismatch:.2e}"
        )
    else:
        lines.append(
            f"Detector energy (factorized): {analysis.detector_energy_au:.4e} Ha "
            f"({analysis.detector_energy_ev:.4e} eV, {analysis.detector_energy_joules:.4e} J)"
        )

    if detector_spectral_energy is not None and factorized_detector_energy is not None:
        ratio_str = (
            f", factorized/exact = {analysis.factorized_to_exact_ratio:.4f}"
            if analysis.factorized_to_exact_ratio is not None
            else ""
        )
        lines.append(
            f"Spectral dU/domega(omega_ref) = {detector_spectral_energy.dU_domega:.4e} a.u., "
            f"Delta_omega_eff = {factorized_detector_energy.effective_bandwidth:.4e} a.u., "
            f"U_fact = {factorized_detector_energy.total_energy_au:.4e} Ha{ratio_str}"
        )

    inc_line = (
        f"Incident laser energy: {analysis.incident_energy_au:.4e} Ha "
        f"({analysis.incident_energy_joules:.4e} J)"
    )
    if analysis.interception_fraction is not None:
        inc_line += (
            f" | bunch interception = {100.0 * analysis.interception_fraction:.2f}%"
        )
    lines.append(inc_line)

    lines.append(
        f"Transfer efficiency: eta_laser = U_det/U_inc = {analysis.laser_energy_fraction:.4e}, "
        f"R_photon = {analysis.photon_reflectivity:.4e} | "
        f"per-photon energy source (D={analysis.doppler_factor:.3f}): "
        f"{100.0 * analysis.laser_energy_share_per_photon:.2f}% laser, "
        f"{100.0 * analysis.electron_energy_share_per_photon:.2f}% electrons"
    )

    if analysis.thin_sheet_benchmark is not None:
        bm = analysis.thin_sheet_benchmark
        ratio_txt = (
            f"{bm.simulated_to_weak_ratio:.4f}"
            if bm.simulated_to_weak_ratio is not None
            else "n/a"
        )
        lines.append(
            f"Thin-sheet theory: alpha = {bm.coupling_alpha:.4e}, "
            f"U_theory(weak) = {bm.weak_theory_energy_au:.4e} Ha, "
            f"U_sim / U_theory = {ratio_txt}"
        )

    if analysis.electron_kinetic_energy is not None:
        ke = analysis.electron_kinetic_energy
        lines.append(
            f"Electron KE: init = {ke.initial_au:.4e} Ha ({ke.initial_kev:.2f} keV, "
            f"{ke.initial_joules:.4e} J), final = {ke.final_au:.4e} Ha, "
            f"Delta_K_e = {ke.delta_au:+.4e} Ha ({ke.delta_joules:+.4e} J)"
        )

    if analysis.larmor_energy_au is not None:
        gain_txt = (
            f", coherence gain U_det/U_Larmor = {analysis.coherence_gain:.3e}"
            if analysis.coherence_gain is not None
            else ""
        )
        lines.append(
            f"Incoherent Larmor emission (4pi): {analysis.larmor_energy_au:.4e} Ha{gain_txt}"
        )

    sat_parts: list[str] = []
    if (
        analysis.saturation_electrons_reflectivity is not None
        and analysis.saturation_charge_pc_reflectivity is not None
    ):
        sat_parts.append(
            f"R_photon={analysis.saturation_threshold:g} at N_e={analysis.saturation_electrons_reflectivity:.3e} "
            f"({analysis.saturation_charge_pc_reflectivity:.2f} pC)"
        )
    if (
        analysis.saturation_electrons_kinetic is not None
        and analysis.saturation_charge_pc_kinetic is not None
    ):
        sat_parts.append(
            f"U_det/K_e={analysis.saturation_threshold:g} at N_e={analysis.saturation_electrons_kinetic:.3e} "
            f"({analysis.saturation_charge_pc_kinetic:.2f} pC)"
        )
    if sat_parts:
        lines.append("Test-particle thresholds: " + "; ".join(sat_parts))

    if analysis.physical_extrapolation is not None:
        pe = analysis.physical_extrapolation
        kin_txt = (
            f", U_det/K_e = {pe.radiated_to_kinetic_ratio:.4e}"
            if pe.radiated_to_kinetic_ratio is not None
            else ""
        )
        valid_txt = (
            f", test-particle valid = {pe.test_particle_regime_valid}"
            if pe.test_particle_regime_valid is not None
            else ""
        )
        lines.append(
            f"Physical bunch extrapolation (N_e={pe.physical_electrons:.3e}, Q={pe.bunch_charge_pc:.3f} pC): "
            f"U_det = {pe.detector_energy_au:.4e} Ha ({pe.detector_energy_joules:.4e} J), "
            f"R_photon = {pe.photon_reflectivity:.4e}{kin_txt}{valid_txt}"
        )

    return "\n".join(lines)
