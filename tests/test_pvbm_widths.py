import numpy as np
import pytest

from vascx_models.metrics.pvbm_widths import measure_pvbm_mask_width


def test_measure_pvbm_mask_width_rectangular_vessel() -> None:
    height = width = 64
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 29:36] = True

    result = measure_pvbm_mask_width(
        vessel_mask=mask,
        center_xy=np.array([32.0, 32.0], dtype=float),
        tangent_xy=np.array([0.0, 1.0], dtype=float),
        max_asymmetry_px=1.0,
    )

    assert result["measurement_valid"] is True
    assert result["width_method"] == "pvbm_mask"
    assert result["width_px"] == pytest.approx(7.0)
    assert result["mask_width_px"] == pytest.approx(7.0)


def test_measure_pvbm_mask_width_rejects_asymmetric_cross_section() -> None:
    height = width = 64
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 20:28] = True

    result = measure_pvbm_mask_width(
        vessel_mask=mask,
        center_xy=np.array([25.0, 32.0], dtype=float),
        tangent_xy=np.array([0.0, 1.0], dtype=float),
        max_asymmetry_px=1.0,
    )

    assert result["measurement_valid"] is False
    assert (
        result["measurement_failure_reason"] == "pvbm_mask_asymmetry_exceeds_threshold"
    )
