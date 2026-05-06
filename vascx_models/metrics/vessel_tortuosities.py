from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import splprep, splev

from ..config import OverlayCircle
from ..geometry.vessel_masks import typed_vessel_masks
from ..geometry.vessel_paths import trace_vessel_tortuosity_paths_between_disc_circle_pair

VESSEL_TORTUOSITY_COLUMNS = [
    "image_id",
    "inner_circle",
    "outer_circle",
    "inner_circle_radius_px",
    "outer_circle_radius_px",
    "connection_index",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "path_length_px",
    "chord_length_px",
    "tortuosity",
    "vessel_type",
]

VESSEL_TORTUOSITY_SUMMARY_COLUMNS = [
    "image_id",
    "metric",
    "vessel_type",
    "inner_circle",
    "outer_circle",
    "inner_circle_radius_px",
    "outer_circle_radius_px",
    "n_segments",
    "n_start_points",
    "total_length_px",
    "mean_tortuosity_weighted",
]

TortuosityMethod = Literal["simple", "curvature"]


def curvature_tortuosity(points: np.ndarray) -> float:
    """
    Compute curvature tortuosity of a 2D centerline using a B-spline.
    """
    points = np.asarray(points, dtype=float)
    eps = 1e-12

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    diffs = np.diff(points, axis=0)
    keep = np.r_[True, np.linalg.norm(diffs, axis=1) > eps]
    points = points[keep]

    n_points = len(points)
    if n_points < 3:
        return 0.0

    spline_degree = min(3, n_points - 1)
    try:
        spline, _ = splprep(
            [points[:, 0], points[:, 1]],
            s=0.0,
            k=spline_degree,
            per=False,
        )
    except ValueError:
        return 0.0

    u = np.linspace(0.0, 1.0, 500)
    dx, dy = splev(u, spline, der=1)
    ddx, ddy = splev(u, spline, der=2)

    dx = np.asarray(dx)
    dy = np.asarray(dy)
    ddx = np.asarray(ddx)
    ddy = np.asarray(ddy)

    speed = np.sqrt(dx**2 + dy**2)
    valid = speed > eps

    kappa = np.zeros_like(speed)
    kappa[valid] = (
        np.abs(dx[valid] * ddy[valid] - dy[valid] * ddx[valid]) / speed[valid] ** 3
    )

    numerator = np.trapezoid(kappa**2 * speed, u)
    path_length = np.trapezoid(speed, u)
    if path_length <= eps:
        return 0.0

    return float(numerator / path_length)


def compute_path_tortuosity(
    path_xy: np.ndarray,
    method: TortuosityMethod = "simple",
) -> tuple[float, float, float]:
    """Return path length, chord length, and tortuosity for an ordered path."""
    if len(path_xy) < 2:
        return 0.0, 0.0, float("nan")

    deltas = np.diff(path_xy, axis=0)
    path_length_px = float(np.hypot(deltas[:, 0], deltas[:, 1]).sum())
    chord_length_px = float(np.linalg.norm(path_xy[-1] - path_xy[0]))
    if method == "simple":
        tortuosity = (
            path_length_px / chord_length_px if chord_length_px > 0.0 else float("nan")
        )
    elif method == "curvature":
        tortuosity = curvature_tortuosity(path_xy)
    else:
        raise ValueError(f"Unsupported vessel tortuosity method: {method}")
    return path_length_px, chord_length_px, float(tortuosity)


def vessel_tortuosity_record(
    image_id: str,
    vessel_type: str,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    inner_radius_px: float,
    outer_radius_px: float,
    connection_index: int,
    path_xy: np.ndarray,
    method: TortuosityMethod = "simple",
) -> dict[str, object]:
    """Build a one-row vessel tortuosity record from an ordered skeleton path."""
    path_length_px, chord_length_px, tortuosity = compute_path_tortuosity(
        path_xy,
        method=method,
    )
    return {
        "image_id": image_id,
        "inner_circle": inner_circle.name,
        "outer_circle": outer_circle.name,
        "inner_circle_radius_px": inner_radius_px,
        "outer_circle_radius_px": outer_radius_px,
        "connection_index": connection_index,
        "x_start": float(path_xy[0, 0]),
        "y_start": float(path_xy[0, 1]),
        "x_end": float(path_xy[-1, 0]),
        "y_end": float(path_xy[-1, 1]),
        "path_length_px": path_length_px,
        "chord_length_px": chord_length_px,
        "tortuosity": tortuosity,
        "vessel_type": vessel_type,
    }


def summarize_vessel_tortuosities(
    df_tortuosities: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Aggregate per-segment tortuosities into length-weighted summaries."""
    if df_tortuosities.empty:
        df_summary = pd.DataFrame(columns=VESSEL_TORTUOSITY_SUMMARY_COLUMNS)
        if output_path is not None:
            df_summary.to_csv(output_path, index=False)
        return df_summary

    group_cols = [
        "image_id",
        "vessel_type",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
    ]
    summary_records: list[dict[str, object]] = []
    for group_key, group in df_tortuosities.groupby(group_cols, dropna=False):
        valid = group.loc[
            np.isfinite(group["tortuosity"])
            & np.isfinite(group["path_length_px"])
            & (group["path_length_px"] > 0.0)
        ]
        (
            image_id,
            vessel_type,
            inner_circle,
            outer_circle,
            inner_radius_px,
            outer_radius_px,
        ) = group_key
        total_length_px = float(valid["path_length_px"].sum())
        n_start_points = int(valid[["x_start", "y_start"]].drop_duplicates().shape[0])
        weighted_tortuosity = float("nan")
        if total_length_px > 0.0:
            weighted_tortuosity = float(
                np.average(valid["tortuosity"], weights=valid["path_length_px"])
            )
        summary_records.append(
            {
                "image_id": image_id,
                "metric": "TORTA" if vessel_type == "artery" else "TORTV",
                "vessel_type": vessel_type,
                "inner_circle": inner_circle,
                "outer_circle": outer_circle,
                "inner_circle_radius_px": float(inner_radius_px),
                "outer_circle_radius_px": float(outer_radius_px),
                "n_segments": int(len(valid)),
                "n_start_points": n_start_points,
                "total_length_px": total_length_px,
                "mean_tortuosity_weighted": weighted_tortuosity,
            }
        )

    df_summary = pd.DataFrame.from_records(
        summary_records,
        columns=VESSEL_TORTUOSITY_SUMMARY_COLUMNS,
    )
    if output_path is not None:
        df_summary.to_csv(output_path, index=False)
    return df_summary


def measure_vessel_tortuosities_between_disc_circle_pair(
    vessels_dir: Path,
    av_dir: Path,
    disc_geometry_path: Path,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    output_path: Optional[Path] = None,
    boundary_tolerance_px: float = 1.5,
    method: TortuosityMethod = "simple",
) -> pd.DataFrame:
    """Measure per-segment vessel tortuosities between two circles."""
    if not disc_geometry_path.exists():
        raise FileNotFoundError(f"Disc geometry file not found: {disc_geometry_path}")
    if not vessels_dir.exists():
        raise FileNotFoundError(f"Vessels directory not found: {vessels_dir}")
    if not av_dir.exists():
        raise FileNotFoundError(f"AV directory not found: {av_dir}")

    df_geometry = pd.read_csv(disc_geometry_path, index_col=0)
    tortuosity_records: list[dict] = []
    for image_id, row in df_geometry.iterrows():
        if (
            pd.isna(row["x_disc_center"])
            or pd.isna(row["y_disc_center"])
            or pd.isna(row["disc_radius_px"])
        ):
            continue

        vessel_path = vessels_dir / f"{image_id}.png"
        av_path = av_dir / f"{image_id}.png"
        if not vessel_path.exists() or not av_path.exists():
            continue

        vessel_mask = np.array(Image.open(vessel_path)) > 0
        av_mask = np.array(Image.open(av_path))
        artery_mask, vein_mask = typed_vessel_masks(vessel_mask, av_mask)
        disc_center_xy = np.array(
            [row["x_disc_center"], row["y_disc_center"]], dtype=float
        )
        inner_radius_px = float(row["disc_radius_px"] * inner_circle.diameter)
        outer_radius_px = float(row["disc_radius_px"] * outer_circle.diameter)

        for typed_mask, vessel_type in ((artery_mask, "artery"), (vein_mask, "vein")):
            vessel_paths = trace_vessel_tortuosity_paths_between_disc_circle_pair(
                vessel_mask=typed_mask,
                disc_center_xy=disc_center_xy,
                inner_radius_px=inner_radius_px,
                outer_radius_px=outer_radius_px,
                boundary_tolerance_px=boundary_tolerance_px,
            )
            tortuosity_records.extend(
                vessel_tortuosity_record(
                    image_id=image_id,
                    vessel_type=vessel_type,
                    inner_circle=inner_circle,
                    outer_circle=outer_circle,
                    inner_radius_px=inner_radius_px,
                    outer_radius_px=outer_radius_px,
                    connection_index=vessel_path.connection_index,
                    path_xy=vessel_path.path_xy,
                    method=method,
                )
                for vessel_path in vessel_paths
            )

    df_tortuosities = pd.DataFrame.from_records(
        tortuosity_records,
        columns=VESSEL_TORTUOSITY_COLUMNS,
    )
    if output_path is not None:
        df_tortuosities.to_csv(output_path, index=False)
    return df_tortuosities
