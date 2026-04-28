from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from .config import OverlayCircle
from .vessel_paths import trace_vessel_tortuosity_paths_between_disc_circle_pair

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


def compute_path_tortuosity(path_xy: np.ndarray) -> tuple[float, float, float]:
    """Return path length, chord length, and tortuosity for an ordered path."""
    if len(path_xy) < 2:
        return 0.0, 0.0, float("nan")

    deltas = np.diff(path_xy, axis=0)
    path_length_px = float(np.hypot(deltas[:, 0], deltas[:, 1]).sum())
    chord_length_px = float(np.linalg.norm(path_xy[-1] - path_xy[0]))
    tortuosity = (
        path_length_px / chord_length_px
        if chord_length_px > 0.0
        else float("nan")
    )
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
) -> dict[str, object]:
    """Build a one-row vessel tortuosity record from an ordered skeleton path."""
    path_length_px, chord_length_px, tortuosity = compute_path_tortuosity(path_xy)
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


def _typed_vessel_masks(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    artery_mask = vessel_mask & np.isin(av_mask, (1, 3))
    vein_mask = vessel_mask & np.isin(av_mask, (2, 3))
    return artery_mask, vein_mask


def measure_vessel_tortuosities_between_disc_circle_pair(
    vessels_dir: Path,
    av_dir: Path,
    disc_geometry_path: Path,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    output_path: Optional[Path] = None,
    boundary_tolerance_px: float = 1.5,
) -> pd.DataFrame:
    """Measure full inner-to-outer vessel tortuosities between two circles."""
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
        artery_mask, vein_mask = _typed_vessel_masks(vessel_mask, av_mask)
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
