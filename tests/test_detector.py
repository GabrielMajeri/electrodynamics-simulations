import matplotlib.pyplot as plt
import numpy as np
import pytest

from electrodynamics.angular_momentum import compute_angular_momentum_derivative
from electrodynamics.detector import (
    DetectorParameters,
    initialize_detector_positions,
    initialize_detector_positions_negative_z,
)
from electrodynamics.plotting import plot_detector_radiation_and_oam


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
