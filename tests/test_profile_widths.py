import numpy as np
import pytest

from vascx_models.metrics.profile_widths import measure_profile_width


def test_measure_profile_width_detects_dark_vertical_vessel() -> None:
    height = width = 64
    channel = np.full((height, width), 0.9, dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    channel[:, 29:36] = 0.1
    mask[:, 29:36] = True

    result = measure_profile_width(
        channel_image=channel,
        vessel_mask=mask,
        center_xy=np.array([32.0, 32.0], dtype=float),
        tangent_xy=np.array([0.0, 1.0], dtype=float),
        channel_name="green",
        half_length_px=12.0,
        sample_step_px=0.25,
        smoothing_sigma_px=1.0,
        threshold_alpha=0.5,
        min_contrast=0.05,
        min_width_px=1.0,
        max_width_px=80.0,
        use_mask_guardrail=True,
        mask_guardrail_min_ratio=0.4,
        mask_guardrail_max_ratio=2.5,
        mask_width_px=7.0,
    )

    assert result["measurement_valid"] is True
    assert result["width_method"] == "profile"
    assert result["profile_channel"] == "green"
    assert result["width_px"] == pytest.approx(7.0, abs=0.75)
    assert result["profile_contrast"] > 0.6


def test_measure_profile_width_rejects_low_contrast_profiles() -> None:
    height = width = 64
    channel = np.full((height, width), 0.505, dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    channel[:, 29:36] = 0.5
    mask[:, 29:36] = True

    result = measure_profile_width(
        channel_image=channel,
        vessel_mask=mask,
        center_xy=np.array([32.0, 32.0], dtype=float),
        tangent_xy=np.array([0.0, 1.0], dtype=float),
        channel_name="green",
        half_length_px=12.0,
        sample_step_px=0.25,
        smoothing_sigma_px=0.0,
        threshold_alpha=0.5,
        min_contrast=0.05,
        min_width_px=1.0,
        max_width_px=80.0,
        use_mask_guardrail=True,
        mask_guardrail_min_ratio=0.4,
        mask_guardrail_max_ratio=2.5,
        mask_width_px=7.0,
    )

    assert result["measurement_valid"] is False
    assert result["measurement_failure_reason"] == "profile_contrast_too_low"


def test_measure_profile_width_rejects_mask_guardrail_violation() -> None:
    height = width = 64
    channel = np.full((height, width), 0.9, dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    channel[:, 25:39] = 0.1
    mask[:, 30:34] = True

    result = measure_profile_width(
        channel_image=channel,
        vessel_mask=mask,
        center_xy=np.array([32.0, 32.0], dtype=float),
        tangent_xy=np.array([0.0, 1.0], dtype=float),
        channel_name="green",
        half_length_px=16.0,
        sample_step_px=0.25,
        smoothing_sigma_px=1.0,
        threshold_alpha=0.5,
        min_contrast=0.05,
        min_width_px=1.0,
        max_width_px=80.0,
        use_mask_guardrail=True,
        mask_guardrail_min_ratio=0.4,
        mask_guardrail_max_ratio=1.2,
        mask_width_px=4.0,
    )

    assert result["measurement_valid"] is False
    assert result["measurement_failure_reason"] == "profile_mask_guardrail_violation"
