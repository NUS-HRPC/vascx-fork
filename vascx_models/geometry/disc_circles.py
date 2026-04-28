import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from ..config import OverlayCircle

logger = logging.getLogger(__name__)


def _save_visual_circle_mask(mask: np.ndarray, path: Path, color: tuple[int, int, int]) -> None:
    image = Image.fromarray(mask.astype(np.uint8))
    palette = [0] * (256 * 3)
    palette[255 * 3 : 255 * 3 + 3] = list(color)
    image.putpalette(palette)
    image.save(path)


def estimate_disc_geometry(
    disc_mask: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Estimate optic disc center and radius from a binary mask."""
    mask = disc_mask > 0
    if not np.any(mask):
        return None

    ys, xs = np.nonzero(mask)
    center_x = float(xs.mean())
    center_y = float(ys.mean())

    # Use the equivalent-circle radius so the estimate is stable for irregular masks.
    radius = float(np.sqrt(mask.sum() / np.pi))
    return center_x, center_y, radius


def create_circle_mask(
    image_shape: Tuple[int, int],
    center: Tuple[float, float],
    radius: float,
    thickness: int,
) -> np.ndarray:
    """Create a binary mask for a circle outline."""
    height, width = image_shape
    circle = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(circle)

    center_x, center_y = center
    bbox = (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )
    draw.ellipse(bbox, outline=255, width=thickness)
    return np.array(circle, dtype=np.uint8)


def generate_disc_circles(
    disc_dir: Path,
    circle_output_dir: Path,
    circles: Sequence[OverlayCircle],
    measurements_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate configurable optic-disc circle masks from saved disc segmentations."""
    circle_output_dir.mkdir(exist_ok=True, parents=True)
    circle_dirs = {circle.name: circle_output_dir / circle.name for circle in circles}
    for circle_dir in circle_dirs.values():
        circle_dir.mkdir(exist_ok=True, parents=True)

    disc_files = list(disc_dir.glob("*.png"))
    if not disc_files:
        logger.warning("No disc masks found for circle generation in %s", disc_dir)
        columns = ["x_disc_center", "y_disc_center", "disc_radius_px"] + [
            f"circle_{circle.name}_px" for circle in circles
        ]
        return pd.DataFrame(columns=columns)

    records: Dict[str, Dict[str, float]] = {}
    logger.info(
        "Generating %d disc circle set(s) for %d disc masks",
        len(circles),
        len(disc_files),
    )

    for disc_file in disc_files:
        image_id = disc_file.stem
        disc_mask = np.array(Image.open(disc_file)) > 0
        geometry = estimate_disc_geometry(disc_mask)

        if geometry is None:
            logger.warning("Disc mask is empty for %s; writing blank circle masks", image_id)
            blank = np.zeros(disc_mask.shape, dtype=np.uint8)
            records[image_id] = {
                "x_disc_center": np.nan,
                "y_disc_center": np.nan,
                "disc_radius_px": np.nan,
            }
            for circle in circles:
                _save_visual_circle_mask(
                    blank,
                    circle_dirs[circle.name] / f"{image_id}.png",
                    circle.color,
                )
                records[image_id][f"circle_{circle.name}_px"] = np.nan
            continue

        center_x, center_y, disc_radius = geometry
        line_width = 1
        records[image_id] = {
            "x_disc_center": center_x,
            "y_disc_center": center_y,
            "disc_radius_px": disc_radius,
        }
        for circle in circles:
            circle_mask = create_circle_mask(
                disc_mask.shape,
                (center_x, center_y),
                radius=disc_radius * circle.diameter,
                thickness=line_width,
            )
            _save_visual_circle_mask(
                circle_mask,
                circle_dirs[circle.name] / f"{image_id}.png",
                circle.color,
            )
            records[image_id][f"circle_{circle.name}_px"] = disc_radius * circle.diameter

    df_measurements = pd.DataFrame.from_dict(records, orient="index")
    if measurements_path is not None:
        df_measurements.to_csv(measurements_path)
        logger.info("Disc circle measurements saved to %s", measurements_path)
    return df_measurements
