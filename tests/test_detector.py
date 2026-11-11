import numpy as np
import pytest

from electrodynamics.detector import (
    DetectorParameters,
    initialize_detector_positions,
    initialize_detector_positions_negative_z,
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
