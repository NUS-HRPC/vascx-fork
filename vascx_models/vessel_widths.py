import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .config import OverlayCircle, VesselWidthConfig
from .profile_widths import measure_profile_width
from .pvbm_widths import measure_pvbm_mask_width
from .vessel_masks import typed_vessel_masks
from .vessel_paths import (
    interpolate_path_point,
    path_cumulative_lengths,
    skeletonize,
    trace_vessel_paths_between_disc_circle_pair,
)

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_DIRECTION_LAG_PX = 6.0

VESSEL_WIDTH_COLUMNS = [
    "image_id",
    "inner_circle",
    "outer_circle",
    "inner_circle_radius_px",
    "outer_circle_radius_px",
    "connection_index",
    "sample_index",
    "x",
    "y",
    "width_px",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "vessel_type",
    "width_method",
    "normal_x",
    "normal_y",
    "profile_channel",
    "profile_left_t",
    "profile_right_t",
    "profile_trough_t",
    "profile_trough_value",
    "profile_background_value",
    "profile_contrast",
    "profile_threshold",
    "profile_confidence",
    "mask_width_px",
    "measurement_valid",
    "measurement_failure_reason",
]


def _nan_point() -> np.ndarray:
    return np.array([float("nan"), float("nan")], dtype=float)


def _base_measurement_result(width_method: str = "mask") -> dict[str, object]:
    return {
        "width_method": width_method,
        "width_px": float("nan"),
        "x_start": float("nan"),
        "y_start": float("nan"),
        "x_end": float("nan"),
        "y_end": float("nan"),
        "normal_x": float("nan"),
        "normal_y": float("nan"),
        "profile_channel": None,
        "profile_left_t": float("nan"),
        "profile_right_t": float("nan"),
        "profile_trough_t": float("nan"),
        "profile_trough_value": float("nan"),
        "profile_background_value": float("nan"),
        "profile_contrast": float("nan"),
        "profile_threshold": float("nan"),
        "profile_confidence": float("nan"),
        "mask_width_px": float("nan"),
        "measurement_valid": False,
        "measurement_failure_reason": None,
    }


def _normalize_measurement_result(
    measurement: dict[str, object],
) -> dict[str, object]:
    normalized = _base_measurement_result(str(measurement.get("width_method", "mask")))
    normalized.update(measurement)
    return normalized


def _local_skeleton_points(
    skeleton: np.ndarray,
    point_xy: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    x, y = point_xy
    height, width = skeleton.shape
    x_min = max(0, int(np.floor(x - radius_px)))
    x_max = min(width - 1, int(np.ceil(x + radius_px)))
    y_min = max(0, int(np.floor(y - radius_px)))
    y_max = min(height - 1, int(np.ceil(y + radius_px)))

    window = skeleton[y_min : y_max + 1, x_min : x_max + 1]
    ys, xs = np.nonzero(window)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=float)

    xs = xs.astype(float) + float(x_min)
    ys = ys.astype(float) + float(y_min)
    deltas = np.column_stack((xs - x, ys - y))
    keep = np.sum(deltas * deltas, axis=1) <= radius_px * radius_px
    return np.column_stack((xs[keep], ys[keep]))


def _estimate_tangent(points_xy: np.ndarray) -> Optional[np.ndarray]:
    if len(points_xy) < 2:
        return None

    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0:
        return None
    return tangent / norm


def _estimate_path_tangent(
    path_xy: np.ndarray,
    cumulative_lengths: np.ndarray,
    target_length: float,
    lag_px: float,
) -> Optional[np.ndarray]:
    if len(path_xy) < 2 or len(cumulative_lengths) < 2:
        return None

    total_length = float(cumulative_lengths[-1])
    if total_length <= 0.0:
        return None

    lower_length = max(0.0, float(target_length - lag_px))
    upper_length = min(total_length, float(target_length + lag_px))
    if upper_length <= lower_length:
        lower_length = 0.0
        upper_length = total_length
    if upper_length <= lower_length:
        return None

    lower_point = interpolate_path_point(path_xy, cumulative_lengths, lower_length)
    upper_point = interpolate_path_point(path_xy, cumulative_lengths, upper_length)
    tangent_xy = upper_point - lower_point
    norm = float(np.linalg.norm(tangent_xy))
    if norm == 0.0:
        return None
    return tangent_xy / norm


def _sample_mask(mask: np.ndarray, point_xy: np.ndarray) -> float:
    x, y = point_xy
    height, width = mask.shape
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    dx = x - x0
    dy = y - y0

    v00 = float(mask[y0, x0])
    v10 = float(mask[y0, x1])
    v01 = float(mask[y1, x0])
    v11 = float(mask[y1, x1])

    top = v00 * (1.0 - dx) + v10 * dx
    bottom = v01 * (1.0 - dx) + v11 * dx
    return top * (1.0 - dy) + bottom * dy


def _trace_boundary_distance(
    vessel_mask: np.ndarray,
    point_xy: np.ndarray,
    direction_xy: np.ndarray,
    step_px: float,
) -> float:
    max_distance = float(np.hypot(*vessel_mask.shape)) + 2.0
    current_value = _sample_mask(vessel_mask, point_xy)
    if current_value < 0.5:
        return float("nan")

    t_inside = 0.0
    steps = int(np.ceil(max_distance / step_px))
    for step in range(1, steps + 1):
        t_outside = step * step_px
        sample_point = point_xy + direction_xy * t_outside
        if _sample_mask(vessel_mask, sample_point) < 0.5:
            low = t_inside
            high = t_outside
            for _ in range(12):
                mid = (low + high) / 2.0
                mid_point = point_xy + direction_xy * mid
                if _sample_mask(vessel_mask, mid_point) >= 0.5:
                    low = mid
                else:
                    high = mid
            return low
        t_inside = t_outside

    return float("nan")


def _boundary_point(
    center_xy: np.ndarray,
    direction_xy: np.ndarray,
    distance_px: float,
) -> np.ndarray:
    return center_xy + direction_xy * distance_px


def _normal_from_tangent(tangent_xy: np.ndarray) -> Optional[np.ndarray]:
    normal_xy = np.array([-tangent_xy[1], tangent_xy[0]], dtype=float)
    norm = float(np.linalg.norm(normal_xy))
    if norm == 0.0:
        return None
    return normal_xy / norm


def _measure_width_along_normal(
    vessel_mask: np.ndarray,
    center_xy: np.ndarray,
    tangent_xy: np.ndarray,
    step_px: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    normal_xy = _normal_from_tangent(tangent_xy)
    if normal_xy is None:
        nan_point = _nan_point()
        return float("nan"), nan_point, nan_point

    positive = _trace_boundary_distance(vessel_mask, center_xy, normal_xy, step_px)
    negative = _trace_boundary_distance(vessel_mask, center_xy, -normal_xy, step_px)
    if np.isnan(positive) or np.isnan(negative):
        nan_point = _nan_point()
        return float("nan"), nan_point, nan_point

    start_xy = _boundary_point(center_xy, -normal_xy, negative)
    end_xy = _boundary_point(center_xy, normal_xy, positive)
    return positive + negative, start_xy, end_xy


def _measure_mask_width_from_tangent(
    vessel_mask: np.ndarray,
    center_xy: np.ndarray,
    tangent_xy: np.ndarray,
    step_px: float,
    width_method: str = "mask",
) -> dict[str, object]:
    measurement = _base_measurement_result(width_method)
    normal_xy = _normal_from_tangent(tangent_xy)
    if normal_xy is None:
        measurement["measurement_failure_reason"] = "normal_unavailable"
        return measurement

    measurement["normal_x"] = float(normal_xy[0])
    measurement["normal_y"] = float(normal_xy[1])
    positive = _trace_boundary_distance(vessel_mask, center_xy, normal_xy, step_px)
    negative = _trace_boundary_distance(vessel_mask, center_xy, -normal_xy, step_px)
    if np.isnan(positive) or np.isnan(negative):
        measurement["measurement_failure_reason"] = "mask_boundary_not_found"
        return measurement

    start_xy = _boundary_point(center_xy, -normal_xy, negative)
    end_xy = _boundary_point(center_xy, normal_xy, positive)
    width_px = positive + negative
    measurement.update(
        {
            "width_px": float(width_px),
            "x_start": float(start_xy[0]),
            "y_start": float(start_xy[1]),
            "x_end": float(end_xy[0]),
            "y_end": float(end_xy[1]),
            "mask_width_px": float(width_px),
            "measurement_valid": True,
            "measurement_failure_reason": None,
        }
    )
    return measurement


def measure_vessel_width_at_coordinate(
    vessel_mask: np.ndarray,
    point_xy: np.ndarray,
    skeleton: Optional[np.ndarray] = None,
    tangent_window_px: float = 10.0,
    measurement_step_px: float = 0.25,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Measure vessel width at an arbitrary coordinate using the local skeleton tangent."""
    skeleton = skeletonize(vessel_mask) if skeleton is None else skeleton
    local_points = _local_skeleton_points(skeleton, point_xy, tangent_window_px)
    tangent_xy = _estimate_tangent(local_points)
    if tangent_xy is None:
        nan_point = _nan_point()
        return float("nan"), nan_point, nan_point

    measurement = _measure_mask_width_from_tangent(
        vessel_mask=vessel_mask,
        center_xy=point_xy,
        tangent_xy=tangent_xy,
        step_px=measurement_step_px,
    )
    if not bool(measurement["measurement_valid"]):
        nan_point = _nan_point()
        return float("nan"), nan_point, nan_point

    start_xy = np.array(
        [float(measurement["x_start"]), float(measurement["y_start"])], dtype=float
    )
    end_xy = np.array(
        [float(measurement["x_end"]), float(measurement["y_end"])], dtype=float
    )
    return float(measurement["width_px"]), start_xy, end_xy


def _channel_index(channel_name: str) -> int:
    channel_indices = {"red": 0, "green": 1, "blue": 2}
    return channel_indices[channel_name]


def _load_profile_channel_image(
    image_path: Path,
    channel_name: str,
) -> np.ndarray:
    rgb_image = np.asarray(Image.open(image_path), dtype=np.float32)
    if rgb_image.ndim != 3 or rgb_image.shape[2] < 3:
        raise ValueError(
            f"Profile width measurement requires an RGB image: {image_path}"
        )
    return rgb_image[:, :, _channel_index(channel_name)] / 255.0


def _measure_sample_width(
    vessel_mask: np.ndarray,
    center_xy: np.ndarray,
    vessel_path_xy: np.ndarray,
    cumulative_lengths: np.ndarray,
    target_length: float,
    width_config: VesselWidthConfig,
    skeleton: np.ndarray,
    tangent_window_px: float,
    measurement_step_px: float,
    profile_channel_image: np.ndarray | None,
) -> dict[str, object]:
    if width_config.method == "mask":
        width_px, start_xy, end_xy = measure_vessel_width_at_coordinate(
            vessel_mask=vessel_mask,
            point_xy=center_xy,
            skeleton=skeleton,
            tangent_window_px=tangent_window_px,
            measurement_step_px=measurement_step_px,
        )
        if np.isnan(width_px):
            measurement = _base_measurement_result("mask")
            measurement["measurement_failure_reason"] = "mask_measurement_failed"
            return measurement
        normal_xy = end_xy - start_xy
        norm = float(np.linalg.norm(normal_xy))
        if norm > 0.0:
            normal_xy = normal_xy / norm
        else:
            normal_xy = _nan_point()
        measurement = _base_measurement_result("mask")
        measurement.update(
            {
                "width_px": float(width_px),
                "x_start": float(start_xy[0]),
                "y_start": float(start_xy[1]),
                "x_end": float(end_xy[0]),
                "y_end": float(end_xy[1]),
                "normal_x": float(normal_xy[0]),
                "normal_y": float(normal_xy[1]),
                "mask_width_px": float(width_px),
                "measurement_valid": True,
            }
        )
        return measurement

    lag_px = (
        width_config.pvbm_mask.direction_lag_px
        if width_config.method == "pvbm_mask"
        else DEFAULT_PROFILE_DIRECTION_LAG_PX
    )
    tangent_xy = _estimate_path_tangent(
        vessel_path_xy,
        cumulative_lengths,
        target_length,
        lag_px=lag_px,
    )
    if tangent_xy is None:
        measurement = _base_measurement_result(width_config.method)
        measurement["measurement_failure_reason"] = "path_tangent_unavailable"
        return measurement

    if width_config.method == "pvbm_mask":
        return _normalize_measurement_result(
            measure_pvbm_mask_width(
                vessel_mask=vessel_mask,
                center_xy=center_xy,
                tangent_xy=tangent_xy,
                max_asymmetry_px=width_config.pvbm_mask.max_asymmetry_px,
            )
        )

    if width_config.method != "profile":
        raise ValueError(f"Unsupported vessel width method: {width_config.method}")

    mask_measurement = _measure_mask_width_from_tangent(
        vessel_mask=vessel_mask,
        center_xy=center_xy,
        tangent_xy=tangent_xy,
        step_px=measurement_step_px,
        width_method="mask",
    )
    if profile_channel_image is None:
        return _normalize_measurement_result(mask_measurement)

    return _normalize_measurement_result(
        measure_profile_width(
            channel_image=profile_channel_image,
            vessel_mask=vessel_mask,
            center_xy=center_xy,
            tangent_xy=tangent_xy,
            channel_name=width_config.profile.channel,
            half_length_px=width_config.profile.half_length_px,
            sample_step_px=width_config.profile.sample_step_px,
            smoothing_sigma_px=width_config.profile.smoothing_sigma_px,
            threshold_alpha=width_config.profile.threshold_alpha,
            min_contrast=width_config.profile.min_contrast,
            min_width_px=width_config.profile.min_width_px,
            max_width_px=width_config.profile.max_width_px,
            use_mask_guardrail=width_config.profile.use_mask_guardrail,
            mask_guardrail_min_ratio=width_config.profile.mask_guardrail_min_ratio,
            mask_guardrail_max_ratio=width_config.profile.mask_guardrail_max_ratio,
            mask_width_px=float(mask_measurement["width_px"]),
        )
    )


def _sample_targets_for_path(
    cumulative_lengths: np.ndarray,
    samples_per_connection: int,
) -> list[tuple[int, float]]:
    if samples_per_connection == 0:
        raise ValueError("samples_per_connection must be non-zero")

    if len(cumulative_lengths) < 3:
        return []

    if samples_per_connection > 0:
        return [
            (
                sample_index,
                float(
                    cumulative_lengths[-1]
                    * (sample_index / (samples_per_connection + 1))
                ),
            )
            for sample_index in range(1, samples_per_connection + 1)
        ]

    return [
        (sample_index, float(target_length))
        for sample_index, target_length in enumerate(cumulative_lengths[1:-1], start=1)
    ]


def _width_records_for_image(
    image_id: str,
    vessel_mask: np.ndarray,
    vessel_type: str,
    disc_center_xy: np.ndarray,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    disc_radius_px: float,
    samples_per_connection: int,
    boundary_tolerance_px: float,
    tangent_window_px: float,
    measurement_step_px: float,
    width_config: VesselWidthConfig,
    profile_channel_image: np.ndarray | None = None,
) -> List[dict]:
    if not np.any(vessel_mask):
        return []

    if samples_per_connection == 0:
        raise ValueError("samples_per_connection must be non-zero")

    inner_radius_px = float(disc_radius_px * inner_circle.diameter)
    outer_radius_px = float(disc_radius_px * outer_circle.diameter)
    if outer_radius_px <= inner_radius_px:
        raise ValueError("outer_circle must have a larger radius than inner_circle")

    vessel_paths = trace_vessel_paths_between_disc_circle_pair(
        vessel_mask=vessel_mask,
        disc_center_xy=disc_center_xy,
        inner_radius_px=inner_radius_px,
        outer_radius_px=outer_radius_px,
        boundary_tolerance_px=boundary_tolerance_px,
    )

    width_records: List[dict] = []
    for vessel_path in vessel_paths:
        cumulative_lengths = path_cumulative_lengths(vessel_path.path_xy)
        sample_targets = _sample_targets_for_path(
            cumulative_lengths=cumulative_lengths,
            samples_per_connection=samples_per_connection,
        )
        if not sample_targets:
            continue
        component_records: List[dict] = []
        for sample_index, target_length in sample_targets:
            center_xy = interpolate_path_point(
                path_xy=vessel_path.path_xy,
                cumulative_lengths=cumulative_lengths,
                target_length=target_length,
            )
            measurement = _measure_sample_width(
                vessel_mask=vessel_mask,
                center_xy=center_xy,
                vessel_path_xy=vessel_path.path_xy,
                cumulative_lengths=cumulative_lengths,
                target_length=target_length,
                width_config=width_config,
                skeleton=vessel_path.skeleton,
                tangent_window_px=tangent_window_px,
                measurement_step_px=measurement_step_px,
                profile_channel_image=profile_channel_image,
            )
            if not bool(measurement["measurement_valid"]):
                # TODO: Recover from local tangent/width failures by re-sampling nearby
                # skeleton points.
                component_records = []
                logger.debug(
                    "Skipping %s connection %d because %s failed at sample %d (%s)",
                    image_id,
                    vessel_path.connection_index,
                    width_config.method,
                    sample_index,
                    measurement["measurement_failure_reason"],
                )
                break

            measurement = _normalize_measurement_result(measurement)
            component_records.append(
                {
                    "image_id": image_id,
                    "inner_circle": inner_circle.name,
                    "outer_circle": outer_circle.name,
                    "inner_circle_radius_px": inner_radius_px,
                    "outer_circle_radius_px": outer_radius_px,
                    "connection_index": vessel_path.connection_index,
                    "sample_index": sample_index,
                    "x": float(center_xy[0]),
                    "y": float(center_xy[1]),
                    "vessel_type": vessel_type,
                    **measurement,
                }
            )

        if len(component_records) != len(sample_targets):
            continue
        width_records.extend(component_records)

    return width_records


def resolve_vessel_width_circle_pair(
    circles: Sequence[OverlayCircle],
    inner_circle_name: str | None = None,
    outer_circle_name: str | None = None,
) -> tuple[OverlayCircle, OverlayCircle]:
    """Select the circle pair used for between-circle vessel width sampling."""
    if len(circles) < 2:
        raise ValueError(
            "At least two overlay circles are required for vessel width sampling"
        )

    circles_by_name = {circle.name: circle for circle in circles}
    if inner_circle_name is not None:
        if inner_circle_name not in circles_by_name:
            raise ValueError(f"Unknown inner circle '{inner_circle_name}'")
        inner_circle = circles_by_name[inner_circle_name]
    else:
        inner_circle = min(circles, key=lambda circle: (circle.diameter, circle.name))

    if outer_circle_name is not None:
        if outer_circle_name not in circles_by_name:
            raise ValueError(f"Unknown outer circle '{outer_circle_name}'")
        outer_circle = circles_by_name[outer_circle_name]
    else:
        remaining_circles = [
            circle for circle in circles if circle.name != inner_circle.name
        ]
        larger_circles = [
            circle
            for circle in remaining_circles
            if circle.diameter > inner_circle.diameter
        ]
        candidates = larger_circles if larger_circles else remaining_circles
        outer_circle = min(
            candidates, key=lambda circle: (circle.diameter, circle.name)
        )

    if outer_circle.diameter <= inner_circle.diameter:
        raise ValueError("outer_circle must have a larger diameter than inner_circle")
    return inner_circle, outer_circle


def measure_vessel_widths_between_disc_circle_pair(
    vessels_dir: Path,
    av_dir: Path,
    disc_geometry_path: Path,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    output_path: Optional[Path] = None,
    samples_per_connection: int = 5,
    boundary_tolerance_px: float = 1.5,
    tangent_window_px: float = 10.0,
    measurement_step_px: float = 0.25,
    width_config: VesselWidthConfig | None = None,
    rgb_dir: Path | None = None,
) -> pd.DataFrame:
    """Measure artery and vein widths separately at interior points between two circles."""
    if not disc_geometry_path.exists():
        raise FileNotFoundError(f"Disc geometry file not found: {disc_geometry_path}")
    if not vessels_dir.exists():
        raise FileNotFoundError(f"Vessels directory not found: {vessels_dir}")
    if not av_dir.exists():
        raise FileNotFoundError(f"AV directory not found: {av_dir}")

    width_config = VesselWidthConfig() if width_config is None else width_config

    df_geometry = pd.read_csv(disc_geometry_path, index_col=0)
    width_records: List[dict] = []

    for image_id, row in df_geometry.iterrows():
        if (
            pd.isna(row["x_disc_center"])
            or pd.isna(row["y_disc_center"])
            or pd.isna(row["disc_radius_px"])
        ):
            logger.warning("Skipping %s because disc geometry is missing", image_id)
            continue

        vessel_path = vessels_dir / f"{image_id}.png"
        av_path = av_dir / f"{image_id}.png"
        if not vessel_path.exists() or not av_path.exists():
            logger.warning("Skipping %s because vessel or AV mask is missing", image_id)
            continue

        vessel_mask = np.array(Image.open(vessel_path)) > 0
        av_mask = np.array(Image.open(av_path))
        profile_channel_image = None
        if width_config.method == "profile":
            if rgb_dir is not None:
                profile_image_path = rgb_dir / f"{image_id}.png"
                if profile_image_path.exists():
                    profile_channel_image = _load_profile_channel_image(
                        profile_image_path,
                        channel_name=width_config.profile.channel,
                    )
                elif not width_config.profile.fallback_to_mask:
                    raise FileNotFoundError(
                        "Profile width measurement requires RGB images in "
                        f"{rgb_dir}. Missing: {profile_image_path.name}. Set "
                        "vessel_widths.profile.fallback_to_mask: true to allow fallback."
                    )
            elif not width_config.profile.fallback_to_mask:
                raise FileNotFoundError(
                    "Profile width measurement requires an RGB image source. Set "
                    "vessel_widths.profile.fallback_to_mask: true to allow fallback."
                )

        artery_mask, vein_mask = typed_vessel_masks(vessel_mask, av_mask)
        for typed_mask, vessel_type in ((artery_mask, "artery"), (vein_mask, "vein")):
            image_width_records = _width_records_for_image(
                image_id=image_id,
                vessel_mask=typed_mask,
                vessel_type=vessel_type,
                disc_center_xy=np.array(
                    [row["x_disc_center"], row["y_disc_center"]], dtype=float
                ),
                inner_circle=inner_circle,
                outer_circle=outer_circle,
                disc_radius_px=float(row["disc_radius_px"]),
                samples_per_connection=samples_per_connection,
                boundary_tolerance_px=boundary_tolerance_px,
                tangent_window_px=tangent_window_px,
                measurement_step_px=measurement_step_px,
                width_config=width_config,
                profile_channel_image=profile_channel_image,
            )
            width_records.extend(image_width_records)

    df_widths = pd.DataFrame.from_records(
        width_records,
        columns=VESSEL_WIDTH_COLUMNS,
    )
    if output_path is not None:
        df_widths.to_csv(output_path, index=False)
        logger.info("Vessel path width measurements saved to %s", output_path)
    return df_widths


def revised_vessel_equivalent(
    diameters: Iterable[float],
    vessel_type: str,
    n_largest: int = 6,
    return_rounds: bool = False,
):
    """
    Compute revised CRAE-6 or CRVE-6 using the Knudtson revised formula.

    Parameters
    ----------
    diameters:
        Vessel diameters in consistent units: pixels, microns, etc.
    vessel_type:
        Accepts 'artery'|'arteriole' or 'vein'|'venule'.
    n_largest:
        Number of largest vessels to use. Standard revised method uses 6.
    return_rounds:
        If True, also return intermediate rounds for inspection.

    Returns
    -------
    float
        Final equivalent vessel diameter.
    """

    coeffs = {
        "arteriole": 0.88,
        "venule": 0.95,
    }

    # Accept synonyms
    if vessel_type in ("artery", "arteriole"):
        vt = "arteriole"
    elif vessel_type in ("vein", "venule"):
        vt = "venule"
    else:
        raise ValueError("vessel_type must be 'arteriole'/'artery' or 'venule'/'vein'.")

    coeff = coeffs[vt]

    values = sorted([float(x) for x in diameters if float(x) > 0.0], reverse=True)[
        :n_largest
    ]

    if len(values) < 2:
        raise ValueError("Need at least two valid vessel diameters.")

    rounds: list[list[float]] = [values.copy()]

    while len(values) > 1:
        values = sorted(values, reverse=True)

        next_values: list[float] = []

        # Pair largest with smallest.
        while len(values) >= 2:
            largest = values.pop(0)
            smallest = values.pop(-1)

            combined = coeff * math.sqrt(largest**2 + smallest**2)
            next_values.append(combined)

        # If odd count remains, carry the middle value forward unchanged.
        if values:
            next_values.append(values[0])

        values = next_values
        rounds.append(values.copy())

    result = float(values[0])

    if return_rounds:
        return result, rounds

    return result


def compute_revised_crx_from_widths(
    df_widths: pd.DataFrame,
    n_largest: int = 6,
    return_rounds: bool = False,
):
    """Aggregate per-connection mean widths and compute revised CRAE/CRVE.

    Parameters
    ----------
    df_widths:
        DataFrame produced by `measure_vessel_widths_between_disc_circle_pair` containing
        per-sample `width_px` values and `connection_index` identifiers.
    n_largest:
        Number of largest vessel averages to consider (default 6).
    return_rounds:
        If True, also return the intermediate rounds from the revised algorithm
        for each image/vessel_type as a mapping.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] or tuple[pd.DataFrame, pd.DataFrame, dict]
        Returns `(df_connections, df_equivalents)` or
        `(df_connections, df_equivalents, rounds_map)` when `return_rounds` is True.
        `df_connections` contains the per-connection mean widths and selection flags.
    """

    cols_conn = [
        "image_id",
        "vessel_type",
        "connection_index",
        "vessel_id",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
        "mean_width_px",
        "n_samples",
        "selected_for_equivalent",
    ]
    cols_equiv = [
        "image_id",
        "metric",
        "vessel_type",
        "requested_n_largest",
        "n_vessels_available",
        "n_vessels_used",
        "vessel_ids_used",
        "mean_widths_used_px",
        "equivalent_px",
    ]

    if df_widths.empty:
        df_conn_empty = pd.DataFrame(columns=cols_conn)
        df_equiv_empty = pd.DataFrame(columns=cols_equiv)
        if return_rounds:
            return df_conn_empty, df_equiv_empty, {}
        return df_conn_empty, df_equiv_empty

    group_cols = [
        "image_id",
        "vessel_type",
        "connection_index",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
    ]

    df_conn = (
        df_widths.groupby(group_cols, dropna=False)
        .agg(mean_width_px=("width_px", "mean"), n_samples=("width_px", "count"))
        .reset_index()
    )
    df_conn["vessel_id"] = df_conn.apply(
        lambda row: f"{row['vessel_type']}_{int(row['connection_index'])}",
        axis=1,
    )
    df_conn["selected_for_equivalent"] = False
    df_conn = df_conn[cols_conn]

    results: list[dict] = []
    rounds_map: dict[tuple[str, str], list[list[float]]] = {}

    for (image_id, vessel_type), group in df_conn.groupby(["image_id", "vessel_type"]):
        top = group.sort_values(
            ["mean_width_px", "connection_index"],
            ascending=[False, True],
        ).head(n_largest)
        df_conn.loc[top.index, "selected_for_equivalent"] = True
        diameters = top["mean_width_px"].tolist()
        metric = "CRAE" if vessel_type == "artery" else "CRVE"
        result = {
            "image_id": image_id,
            "metric": metric,
            "vessel_type": vessel_type,
            "requested_n_largest": int(n_largest),
            "n_vessels_available": int(len(group)),
            "n_vessels_used": int(len(top)),
            "vessel_ids_used": ";".join(str(value) for value in top["vessel_id"]),
            "mean_widths_used_px": ";".join(
                f"{float(value):.6g}" for value in diameters
            ),
            "equivalent_px": float("nan"),
        }
        try:
            if return_rounds:
                eq, rounds = revised_vessel_equivalent(
                    diameters,
                    vessel_type,
                    n_largest=n_largest,
                    return_rounds=True,
                )
                rounds_map[(image_id, vessel_type)] = rounds
            else:
                eq = revised_vessel_equivalent(
                    diameters, vessel_type, n_largest=n_largest
                )
            result["equivalent_px"] = float(eq)
        except ValueError:
            pass
        results.append(result)

    df_equiv = pd.DataFrame.from_records(results, columns=cols_equiv)

    if return_rounds:
        return df_conn, df_equiv, rounds_map

    return df_conn, df_equiv


def select_vessel_width_measurements_for_equivalents(
    df_widths: pd.DataFrame,
    df_connections: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-sample width rows whose vessels were selected for CRAE/CRVE."""
    if df_widths.empty or df_connections.empty:
        return df_widths.iloc[0:0].copy()

    selected = df_connections[df_connections["selected_for_equivalent"]]
    if selected.empty:
        return df_widths.iloc[0:0].copy()

    merge_cols = [
        "image_id",
        "vessel_type",
        "connection_index",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
    ]
    return df_widths.merge(selected[merge_cols], on=merge_cols, how="inner")
