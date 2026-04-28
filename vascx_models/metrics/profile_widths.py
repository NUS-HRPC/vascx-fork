from __future__ import annotations

import math

import numpy as np


def _normal_from_tangent(tangent_xy: np.ndarray) -> np.ndarray | None:
    normal_xy = np.array([-tangent_xy[1], tangent_xy[0]], dtype=float)
    norm = float(np.linalg.norm(normal_xy))
    if norm == 0.0:
        return None
    return normal_xy / norm


def _sample_image(image: np.ndarray, point_xy: np.ndarray) -> float:
    x, y = float(point_xy[0]), float(point_xy[1])
    height, width = image.shape
    if x < 0.0 or y < 0.0 or x > width - 1 or y > height - 1:
        return float("nan")

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0

    v00 = float(image[y0, x0])
    v10 = float(image[y0, x1])
    v01 = float(image[y1, x0])
    v11 = float(image[y1, x1])
    top = v00 * (1.0 - dx) + v10 * dx
    bottom = v01 * (1.0 - dx) + v11 * dx
    return top * (1.0 - dy) + bottom * dy


def _gaussian_kernel1d(sigma_samples: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma_samples)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel_sum = float(kernel.sum())
    if kernel_sum == 0.0:
        return np.array([1.0], dtype=float)
    return kernel / kernel_sum


def _smooth_profile(
    profile_values: np.ndarray, sigma_px: float, sample_step_px: float
) -> np.ndarray:
    if sigma_px <= 0.0:
        return profile_values.copy()

    sigma_samples = sigma_px / sample_step_px
    if sigma_samples <= 0.0:
        return profile_values.copy()

    kernel = _gaussian_kernel1d(sigma_samples)
    return np.convolve(profile_values, kernel, mode="same")


def _crossing_position(
    t0: float,
    value0: float,
    t1: float,
    value1: float,
    threshold: float,
) -> float:
    delta = value1 - value0
    if delta == 0.0:
        return float((t0 + t1) / 2.0)
    fraction = (threshold - value0) / delta
    return float(t0 + fraction * (t1 - t0))


def _find_left_crossing(
    t_values: np.ndarray,
    profile_values: np.ndarray,
    trough_index: int,
    threshold: float,
) -> float:
    for index in range(trough_index, 0, -1):
        outer_value = float(profile_values[index - 1])
        inner_value = float(profile_values[index])
        if np.isnan(outer_value) or np.isnan(inner_value):
            continue
        if outer_value >= threshold >= inner_value:
            return _crossing_position(
                float(t_values[index - 1]),
                outer_value,
                float(t_values[index]),
                inner_value,
                threshold,
            )
    return float("nan")


def _find_right_crossing(
    t_values: np.ndarray,
    profile_values: np.ndarray,
    trough_index: int,
    threshold: float,
) -> float:
    for index in range(trough_index, len(profile_values) - 1):
        inner_value = float(profile_values[index])
        outer_value = float(profile_values[index + 1])
        if np.isnan(inner_value) or np.isnan(outer_value):
            continue
        if inner_value <= threshold <= outer_value:
            return _crossing_position(
                float(t_values[index]),
                inner_value,
                float(t_values[index + 1]),
                outer_value,
                threshold,
            )
    return float("nan")


def _profile_background(profile_values: np.ndarray) -> float:
    edge_count = max(3, int(math.ceil(len(profile_values) * 0.1)))
    left_edge = profile_values[:edge_count]
    right_edge = profile_values[-edge_count:]
    edge_values = np.concatenate((left_edge, right_edge))
    finite_values = edge_values[np.isfinite(edge_values)]
    if len(finite_values) == 0:
        return float("nan")
    return float(finite_values.mean())


def measure_profile_width(
    channel_image: np.ndarray,
    vessel_mask: np.ndarray,
    center_xy: np.ndarray,
    tangent_xy: np.ndarray,
    *,
    channel_name: str,
    half_length_px: float,
    sample_step_px: float,
    smoothing_sigma_px: float,
    threshold_alpha: float,
    min_contrast: float,
    min_width_px: float,
    max_width_px: float,
    use_mask_guardrail: bool,
    mask_guardrail_min_ratio: float,
    mask_guardrail_max_ratio: float,
    mask_width_px: float,
) -> dict[str, object]:
    normal_xy = _normal_from_tangent(tangent_xy)
    result = {
        "width_method": "profile",
        "width_px": float("nan"),
        "x_start": float("nan"),
        "y_start": float("nan"),
        "x_end": float("nan"),
        "y_end": float("nan"),
        "normal_x": float("nan"),
        "normal_y": float("nan"),
        "profile_channel": channel_name,
        "profile_left_t": float("nan"),
        "profile_right_t": float("nan"),
        "profile_trough_t": float("nan"),
        "profile_trough_value": float("nan"),
        "profile_background_value": float("nan"),
        "profile_contrast": float("nan"),
        "profile_threshold": float("nan"),
        "profile_confidence": float("nan"),
        "mask_width_px": float(mask_width_px),
        "measurement_valid": False,
        "measurement_failure_reason": None,
    }
    if normal_xy is None:
        result["measurement_failure_reason"] = "normal_unavailable"
        return result

    result["normal_x"] = float(normal_xy[0])
    result["normal_y"] = float(normal_xy[1])

    t_values = np.arange(
        -half_length_px, half_length_px + sample_step_px, sample_step_px
    )
    sample_points = center_xy[None, :] + t_values[:, None] * normal_xy[None, :]
    profile_values = np.asarray(
        [_sample_image(channel_image, point_xy) for point_xy in sample_points],
        dtype=float,
    )
    if not np.isfinite(profile_values).any():
        result["measurement_failure_reason"] = "profile_sampling_failed"
        return result

    smoothed_profile = _smooth_profile(
        np.nan_to_num(profile_values, nan=float(np.nanmean(profile_values))),
        sigma_px=smoothing_sigma_px,
        sample_step_px=sample_step_px,
    )
    trough_index = int(np.argmin(smoothed_profile))
    trough_t = float(t_values[trough_index])
    trough_value = float(smoothed_profile[trough_index])
    background_value = _profile_background(smoothed_profile)
    contrast = background_value - trough_value

    result["profile_trough_t"] = trough_t
    result["profile_trough_value"] = trough_value
    result["profile_background_value"] = background_value
    result["profile_contrast"] = contrast
    if np.isnan(background_value) or contrast < min_contrast:
        result["measurement_failure_reason"] = "profile_contrast_too_low"
        return result

    threshold = trough_value + threshold_alpha * contrast
    left_t = _find_left_crossing(t_values, smoothed_profile, trough_index, threshold)
    right_t = _find_right_crossing(t_values, smoothed_profile, trough_index, threshold)
    result["profile_left_t"] = left_t
    result["profile_right_t"] = right_t
    result["profile_threshold"] = threshold
    if np.isnan(left_t) or np.isnan(right_t) or right_t <= left_t:
        result["measurement_failure_reason"] = "profile_boundary_not_found"
        return result

    width_px = float(right_t - left_t)
    if width_px < min_width_px or width_px > max_width_px:
        result["measurement_failure_reason"] = "profile_width_out_of_range"
        return result

    if use_mask_guardrail:
        if not np.isfinite(mask_width_px) or mask_width_px <= 0.0:
            result["measurement_failure_reason"] = "mask_guardrail_unavailable"
            return result
        width_ratio = width_px / float(mask_width_px)
        if (
            width_ratio < mask_guardrail_min_ratio
            or width_ratio > mask_guardrail_max_ratio
        ):
            result["measurement_failure_reason"] = "profile_mask_guardrail_violation"
            return result

    start_xy = center_xy + normal_xy * left_t
    end_xy = center_xy + normal_xy * right_t
    center_offset = abs(float((left_t + right_t) / 2.0 - trough_t))
    confidence = 1.0 - min(1.0, center_offset / max(width_px / 2.0, 1e-6))
    result.update(
        {
            "width_px": width_px,
            "x_start": float(start_xy[0]),
            "y_start": float(start_xy[1]),
            "x_end": float(end_xy[0]),
            "y_end": float(end_xy[1]),
            "profile_confidence": float(np.clip(confidence, 0.0, 1.0)),
            "measurement_valid": True,
            "measurement_failure_reason": None,
        }
    )
    return result
