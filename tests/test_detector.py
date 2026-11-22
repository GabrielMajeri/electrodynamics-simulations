import matplotlib.pyplot as plt
import numpy as np
import pytest

from electrodynamics.angular_momentum import (
    compute_angular_momentum_derivative,
    compute_particle_angular_momentum_z,
)
from electrodynamics.detector import (
    DetectorParameters,
    initialize_detector_positions,
    initialize_detector_positions_negative_z,
)
from electrodynamics.plotting import (
    plot_detector_field_map,
    plot_detector_radiation_and_oam,
)


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
    from electrodynamics.angular_momentum import compute_angular_momentum_derivative

    params = DetectorParameters(width=20.0, height=20.0, grid_size_x=16, grid_size_y=16)
    pos_flat = initialize_detector_positions_negative_z(params, 100.0)
    pos_2d = pos_flat.reshape(params.grid_size_x, params.grid_size_y, 3)

    omega = 0.057
    x = pos_2d[:, :, 0]
    y = pos_2d[:, :, 1]
    r2 = x**2 + y**2

    # Case 1: Axisymmetric field -> zero net orbital angular momentum
    field_axisym = np.zeros_like(pos_2d, dtype=np.complex128)
    field_axisym[:, :, 0] = np.exp(-r2 / 10.0)
    dL_z_sym = compute_angular_momentum_derivative(pos_2d, field_axisym, omega)
    assert np.abs(np.sum(np.real(dL_z_sym))) < 1e-12

    # Case 2: Vortex field (l = 1) -> non-zero radiated orbital angular momentum
    field_vortex = np.zeros_like(pos_2d, dtype=np.complex128)
    field_vortex[:, :, 0] = (x + 1j * y) * np.exp(-r2 / 10.0)
    dL_z_vortex = compute_angular_momentum_derivative(pos_2d, field_vortex, omega)
    assert np.abs(np.sum(np.real(dL_z_vortex))) > 1.0


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
