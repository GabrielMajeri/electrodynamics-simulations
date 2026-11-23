import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import eval_genlaguerre

from electrodynamics.angular_momentum import (
    compute_angular_momentum_derivative,
    compute_detector_oam_spectral_density,
    compute_oam_per_photon,
    compute_particle_angular_momentum_z,
    integrate_detector_oam_spectral_density,
)
from electrodynamics.constants import SPEED_OF_LIGHT
from electrodynamics.detector import (
    DetectorParameters,
    DiskDetectorParameters,
    build_detector_grid,
    circular_aperture_mask,
    compute_detector_area_weights,
    compute_disk_radial_quadrature,
    create_detector_parameters,
    initialize_detector_positions,
    initialize_detector_positions_negative_z,
)
from electrodynamics.plotting import (
    plot_detector_field_map,
    plot_detector_radiation_and_oam,
    plot_detector_scalar_panel,
    plot_experimental_setup,
    save_detector_field_component_plots,
)
from electrodynamics.serialization import save_json


def test_detector_positions_generation() -> None:
    params = DetectorParameters(width=100.0, height=80.0, grid_size_x=10, grid_size_y=8)
    z_dist = -500.0
    positions = initialize_detector_positions_negative_z(params, z_dist)

    assert positions.shape == (80, 3)
    np.testing.assert_allclose(positions[:, 2], z_dist)
    assert positions[:, 0].min() == pytest.approx(-50.0)
    assert positions[:, 0].max() == pytest.approx(50.0)
    assert positions[:, 1].min() == pytest.approx(-40.0)
    assert positions[:, 1].max() == pytest.approx(40.0)

    # General detector position generator
    pos_arbitrary = initialize_detector_positions(
        params, np.array([0.0, 0.0, z_dist]), np.zeros(3)
    )
    np.testing.assert_allclose(pos_arbitrary, positions)


def test_angular_momentum_derivative() -> None:
    params = DetectorParameters(width=20.0, height=20.0, grid_size_x=64, grid_size_y=64)
    pos_flat = initialize_detector_positions_negative_z(params, -100.0)
    pos_2d = pos_flat.reshape(params.grid_size_x, params.grid_size_y, 3)

    omega = 0.057
    x = pos_2d[:, :, 0]
    y = pos_2d[:, :, 1]
    r2 = x**2 + y**2
    w2 = 2.5**2

    # Case 1: Axisymmetric field -> zero net orbital angular momentum
    field_axisym = np.zeros_like(pos_2d, dtype=np.complex128)
    field_axisym[:, :, 0] = np.exp(-r2 / w2)
    dL_z_sym = compute_angular_momentum_derivative(pos_2d, field_axisym, omega)
    assert abs(integrate_detector_oam_spectral_density(pos_2d, dL_z_sym)) < 1e-12
    assert abs(compute_oam_per_photon(pos_2d, field_axisym, dL_z_sym, omega)) < 1e-12

    # Case 2: Vortex field (l = +1) -> positive lab-frame L_z and +1 OAM per photon
    field_vortex = np.zeros_like(pos_2d, dtype=np.complex128)
    field_vortex[:, :, 0] = (x + 1j * y) * np.exp(-r2 / w2)
    dL_z_vortex = compute_angular_momentum_derivative(pos_2d, field_vortex, omega)
    assert integrate_detector_oam_spectral_density(pos_2d, dL_z_vortex) > 0.0
    assert compute_oam_per_photon(
        pos_2d, field_vortex, dL_z_vortex, omega, propagation_sign=1
    ) == pytest.approx(1.0, rel=1e-2)
    assert compute_oam_per_photon(
        pos_2d, field_vortex, dL_z_vortex, omega, propagation_sign=-1
    ) == pytest.approx(-1.0, rel=1e-2)

    # Case 3: Vortex field (l = -2) with spherical wavefront phase -> -2 OAM per photon
    r_det = np.linalg.norm(pos_2d, axis=-1)
    field_l_minus_2 = np.zeros_like(pos_2d, dtype=np.complex128)
    field_l_minus_2[:, :, 0] = (
        (x - 1j * y) ** 2
        * np.exp(-r2 / w2)
        * np.exp(1j * (omega / SPEED_OF_LIGHT) * r_det)
    )
    dL_z_minus_2 = compute_angular_momentum_derivative(pos_2d, field_l_minus_2, omega)
    assert compute_oam_per_photon(
        pos_2d, field_l_minus_2, dL_z_minus_2, omega, propagation_sign=1
    ) == pytest.approx(-2.0, rel=2e-2)


def test_plot_detector_radiation_and_oam() -> None:
    params = DetectorParameters(width=20.0, height=20.0, grid_size_x=16, grid_size_y=16)
    pos_flat = initialize_detector_positions_negative_z(params, 100.0)
    pos_2d = pos_flat.reshape(params.grid_size_x, params.grid_size_y, 3)

    omega = 0.057
    x = pos_2d[:, :, 0]
    y = pos_2d[:, :, 1]
    r2 = x**2 + y**2

    field = np.zeros_like(pos_2d, dtype=np.complex128)
    field[:, :, 0] = (x + 1j * y) * np.exp(-r2 / 10.0)
    dL_z = compute_angular_momentum_derivative(pos_2d, field, omega)

    fig = plt.figure()
    plot_detector_radiation_and_oam(fig, pos_2d, field, dL_z, laser_wavelength=100.0)
    plt.close(fig)


def test_detector_reshape_to_grid_non_square() -> None:
    params = DetectorParameters(width=120.0, height=60.0, grid_size_x=12, grid_size_y=7)
    pos_flat = initialize_detector_positions_negative_z(params, -100.0)
    grid = params.reshape_to_grid(pos_flat)

    assert grid.shape == (12, 7, 3)
    # x varies along axis 0 and is constant along axis 1
    np.testing.assert_allclose(grid[:, 0, 0], np.linspace(-60.0, 60.0, 12))
    np.testing.assert_allclose(grid[0, :, 1], np.linspace(-30.0, 30.0, 7))

    fig = plt.figure()
    plot_detector_field_map(
        fig, grid[:, :, 0], (-60.0, 60.0, -30.0, 30.0), title="$E_x$"
    )
    plt.close(fig)


def test_particle_angular_momentum_z() -> None:
    pos_3d = np.array([[2.0, 3.0, 0.0], [-1.0, 4.0, 5.0]])
    mom_3d = np.array([[1.0, 5.0, 0.0], [2.0, -3.0, 1.0]])
    lz_3d = compute_particle_angular_momentum_z(pos_3d, mom_3d)
    np.testing.assert_allclose(lz_3d, [2.0 * 5.0 - 3.0 * 1.0, -1.0 * -3.0 - 4.0 * 2.0])

    pos_4d = np.column_stack([np.zeros(2), pos_3d])
    mom_4d = np.column_stack([np.ones(2), mom_3d])
    lz_4d = compute_particle_angular_momentum_z(pos_4d, mom_4d)
    np.testing.assert_allclose(lz_4d, lz_3d)


def _eval_orthonormal_lg_mode(
    x: np.ndarray, y: np.ndarray, waist: float, p: int, l: int
) -> np.ndarray:
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    abs_l = abs(l)
    norm = math.sqrt(2.0 * math.factorial(p) / (math.pi * math.factorial(p + abs_l))) / waist
    xi = 2.0 * (r / waist) ** 2
    return (
        norm
        * ((math.sqrt(2.0) * r / waist) ** abs_l)
        * eval_genlaguerre(p, abs_l, xi)
        * np.exp(-(r**2) / (waist**2))
        * np.exp(1j * l * phi)
    )


def test_disk_detector_quadrature_and_geometry(tmp_path: Path) -> None:
    radius = 25.0
    for sampling in ("gauss-legendre", "uniform", "equal-area"):
        params = DiskDetectorParameters(
            radius=radius,
            num_radial=32,
            num_azimuthal=48,
            radial_sampling=sampling,
        )
        radii, _, angles = compute_disk_radial_quadrature(params)
        assert radii.shape == (32,)
        assert angles.shape == (48,)
        assert np.all((radii > 0.0) & (radii <= radius))
        weights = compute_detector_area_weights(params)
        assert weights.shape == (32, 48)
        np.testing.assert_allclose(np.sum(weights), math.pi * radius**2, rtol=1e-12)

    # High-accuracy Gauss-Legendre LG mode normalization and orthogonality
    gl_params = DiskDetectorParameters(
        radius=radius, num_radial=48, num_azimuthal=64, radial_sampling="gauss-legendre"
    )
    grid = build_detector_grid(
        gl_params, distance=500.0, reflection_angle=math.pi / 6.0, direction=-1
    )
    assert grid.is_polar
    assert grid.grid_shape == (48, 64)

    x_loc = grid.local_coords_2d[..., 0]
    y_loc = grid.local_coords_2d[..., 1]
    gauss = (1.0 / (math.sqrt(math.pi / 2.0) * 3.0)) * np.exp(
        -(x_loc**2 + y_loc**2) / (3.0**2)
    )
    np.testing.assert_allclose(
        np.sum(np.abs(gauss) ** 2 * grid.area_weights_2d), 1.0, atol=1e-11
    )

    modes = [
        _eval_orthonormal_lg_mode(x_loc, y_loc, waist=3.0, p=p, l=l)
        for p, l in ((0, 1), (1, -2), (2, 2))
    ]
    for mode in modes:
        np.testing.assert_allclose(
            np.sum(np.abs(mode) ** 2 * grid.area_weights_2d), 1.0, atol=1e-11
        )
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            overlap = np.sum(np.conj(modes[i]) * modes[j] * grid.area_weights_2d)
            assert abs(overlap) < 1e-11

    # Oblique geometry checks: positions lie in plane perpendicular to normal_lab
    center = 500.0 * grid.normal_lab
    rel_lab = grid.lab_positions_flat - center
    np.testing.assert_allclose(rel_lab @ grid.normal_lab, 0.0, atol=1e-11)
    np.testing.assert_allclose(
        np.linalg.norm(rel_lab, axis=-1),
        np.hypot(x_loc, y_loc).reshape(-1),
        rtol=1e-12,
        atol=1e-12,
    )
    reshaped = grid.reshape_to_grid(grid.lab_positions_flat)
    assert reshaped.shape == (48, 64, 3)

    # Forward direction=+1 grid
    fwd_grid = build_detector_grid(
        gl_params, distance=300.0, reflection_angle=0.2, direction=1
    )
    fwd_center = 300.0 * fwd_grid.normal_lab
    np.testing.assert_allclose(
        (fwd_grid.lab_positions_flat - fwd_center) @ fwd_grid.normal_lab, 0.0, atol=1e-11
    )
    np.testing.assert_allclose(fwd_grid.normal_local, [0.0, 0.0, 1.0])

    # Rectangular area weights & circular aperture mask
    rect_params = DetectorParameters(
        width=40.0, height=30.0, grid_size_x=33, grid_size_y=25
    )
    rect_grid = build_detector_grid(rect_params, distance=100.0)
    assert not rect_grid.is_polar
    np.testing.assert_allclose(np.sum(rect_grid.area_weights_2d), 40.0 * 30.0, rtol=1e-12)
    mask = circular_aperture_mask(rect_grid.local_coords_2d, radius=10.0)
    assert mask.shape == (33, 25)
    assert np.all(
        np.hypot(rect_grid.local_coords_2d[mask, 0], rect_grid.local_coords_2d[mask, 1])
        <= 10.0
    )

    # Serialization check
    save_json(tmp_path / "disk_det.json", {"detector": gl_params})
    assert (tmp_path / "disk_det.json").exists()


def test_polar_oam_spectral_density_and_per_photon() -> None:
    params = DiskDetectorParameters(
        radius=30.0, num_radial=48, num_azimuthal=64, radial_sampling="gauss-legendre"
    )
    grid = build_detector_grid(params, distance=500.0, direction=-1)
    x = grid.local_coords_2d[:, :, 0]
    y = grid.local_coords_2d[:, :, 1]
    r_det = np.linalg.norm(grid.local_coords_2d, axis=-1)
    omega = 50.0
    spherical_carrier = np.exp(1j * (omega / SPEED_OF_LIGHT) * r_det)

    for p, l in ((0, 0), (0, 1), (1, -2), (2, 3)):
        field = np.zeros((*grid.grid_shape, 3), dtype=np.complex128)
        field[:, :, 0] = _eval_orthonormal_lg_mode(x, y, waist=4.0, p=p, l=l) * spherical_carrier
        dL_z = compute_detector_oam_spectral_density(grid, field, omega)
        oam_lab = compute_oam_per_photon(
            grid.local_coords_2d,
            field,
            dL_z,
            omega,
            propagation_sign=1,
            area_weights=grid.area_weights_2d,
        )
        oam_beam = compute_oam_per_photon(
            grid.local_coords_2d,
            field,
            dL_z,
            omega,
            propagation_sign=-1,
            area_weights=grid.area_weights_2d,
        )
        assert oam_lab == pytest.approx(float(l), abs=1e-5)
        assert oam_beam == pytest.approx(float(-l), abs=1e-5)


def test_disk_detector_plotting_smoke(tmp_path: Path) -> None:
    params = DiskDetectorParameters(
        radius=20.0, num_radial=24, num_azimuthal=32, radial_sampling="gauss-legendre"
    )
    grid = build_detector_grid(params, distance=200.0, direction=-1)
    x = grid.local_coords_2d[:, :, 0]
    y = grid.local_coords_2d[:, :, 1]
    omega = 0.057

    field = np.zeros((*grid.grid_shape, 3), dtype=np.complex128)
    field[:, :, 0] = _eval_orthonormal_lg_mode(x, y, waist=3.5, p=0, l=1)
    dL_z = compute_detector_oam_spectral_density(grid, field, omega)

    # 3D setup plot
    fig_setup = plt.figure()
    particle_pos = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]])
    particle_mom = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    plot_experimental_setup(
        fig_setup, particle_pos, particle_mom, params, grid.lab_positions_flat
    )
    plt.close(fig_setup)

    # 3-panel radiation & OAM plot (with and without wave_number)
    for wn in (None, 0.1):
        fig_rad = plt.figure()
        plot_detector_radiation_and_oam(
            fig_rad,
            grid.local_coords_2d,
            field,
            dL_z,
            laser_wavelength=10.0,
            wave_number=wn,
            area_weights=grid.area_weights_2d,
            detector_parameters=params,
        )
        plt.close(fig_rad)

    # Polar single field map
    fig_map = plt.figure()
    plot_detector_field_map(
        fig_map,
        np.real(field[:, :, 0]),
        extent=(-5.0, 5.0, -5.0, 5.0),
        title="$E_x$",
        local_coords_2d=grid.local_coords_2d,
        scale=4.0,
        disk_radius=params.radius,
    )
    plt.close(fig_map)

    # Save component plots
    save_detector_field_component_plots(
        tmp_path,
        field,
        params,
        waist_radius=4.0,
        formats=("png", "pdf"),
        local_coords_2d=grid.local_coords_2d,
    )
    for stem in ("electric_field_x", "electric_field_y", "electric_field_z", "electric_field_norm"):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()


def test_create_detector_parameters_and_scalar_panel() -> None:
    rect = create_detector_parameters("rect", span=100.0, grid_size=16)
    assert isinstance(rect, DetectorParameters)
    assert rect.width == pytest.approx(100.0)
    assert rect.height == pytest.approx(100.0)
    assert rect.grid_size_x == 16 and rect.grid_size_y == 16

    disk_default = create_detector_parameters("disk", span=100.0, grid_size=32)
    assert isinstance(disk_default, DiskDetectorParameters)
    assert disk_default.radius == pytest.approx(50.0)
    assert disk_default.num_radial == 16
    assert disk_default.num_azimuthal == 32

    disk_explicit = create_detector_parameters(
        "disk",
        span=80.0,
        num_radial=20,
        num_azimuthal=40,
        radial_sampling="equal-area",
    )
    assert isinstance(disk_explicit, DiskDetectorParameters)
    assert disk_explicit.radius == pytest.approx(40.0)
    assert disk_explicit.num_radial == 20
    assert disk_explicit.num_azimuthal == 40
    assert disk_explicit.radial_sampling == "equal-area"

    with pytest.raises(ValueError, match="Unknown detector shape"):
        create_detector_parameters("hex", span=10.0)  # pyright: ignore[reportArgumentType]

    for params in (rect, disk_explicit):
        grid = build_detector_grid(params, distance=200.0)
        vals = np.hypot(grid.local_coords_2d[..., 0], grid.local_coords_2d[..., 1])
        fig, ax = plt.subplots()
        plot_detector_scalar_panel(
            ax, grid.local_coords_2d, vals, scale=10.0, detector_parameters=params
        )
        plt.close(fig)
