import logging
from collections import deque
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from ..config import OverlayConfig
from ..geometry.vessel_paths import skeletonize

logger = logging.getLogger(__name__)

_NEIGHBORS_8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _rasterize_line_segments(
    image_shape: tuple[int, int],
    measurements: Sequence[Mapping[str, object]],
) -> np.ndarray:
    mask_image = Image.new("L", (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(mask_image)
    for measurement in measurements:
        start = (
            float(measurement["x_start"]),
            float(measurement["y_start"]),
        )
        end = (
            float(measurement["x_end"]),
            float(measurement["y_end"]),
        )
        draw.line([start, end], fill=255, width=1)
    return np.array(mask_image, dtype=np.uint8) > 0


def _rasterize_vessel_width_measurements(
    image_shape: tuple[int, int],
    measurements: Sequence[Mapping[str, object]],
    vessel_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    rasterized_mask = _rasterize_line_segments(image_shape, measurements)
    if vessel_mask is not None:
        return rasterized_mask & vessel_mask
    return rasterized_mask


def _nearest_skeleton_coordinate(
    skeleton_mask: np.ndarray,
    x_value: float,
    y_value: float,
) -> tuple[int, int] | None:
    rows, cols = np.nonzero(skeleton_mask)
    if len(rows) == 0:
        return None
    deltas = (cols.astype(float) - float(x_value)) ** 2 + (
        rows.astype(float) - float(y_value)
    ) ** 2
    nearest_index = int(np.argmin(deltas))
    return int(rows[nearest_index]), int(cols[nearest_index])


def _trace_skeleton_segment(
    skeleton_mask: np.ndarray,
    start_yx: tuple[int, int],
    end_yx: tuple[int, int],
) -> np.ndarray:
    if start_yx == end_yx:
        mask = np.zeros_like(skeleton_mask, dtype=bool)
        mask[start_yx] = True
        return mask

    parents: dict[tuple[int, int], tuple[int, int] | None] = {start_yx: None}
    queue: deque[tuple[int, int]] = deque([start_yx])

    while queue:
        y_value, x_value = queue.popleft()
        for delta_y, delta_x in _NEIGHBORS_8:
            next_y = y_value + delta_y
            next_x = x_value + delta_x
            next_coord = (next_y, next_x)
            if (
                next_y < 0
                or next_x < 0
                or next_y >= skeleton_mask.shape[0]
                or next_x >= skeleton_mask.shape[1]
                or not skeleton_mask[next_y, next_x]
                or next_coord in parents
            ):
                continue
            parents[next_coord] = (y_value, x_value)
            if next_coord == end_yx:
                queue.clear()
                break
            queue.append(next_coord)

    if end_yx not in parents:
        return np.zeros_like(skeleton_mask, dtype=bool)

    segment_mask = np.zeros_like(skeleton_mask, dtype=bool)
    current = end_yx
    while current is not None:
        segment_mask[current] = True
        current = parents[current]
    return segment_mask


def _rasterize_tortuosity_skeleton_segments(
    vessel_mask: np.ndarray,
    measurements: Sequence[Mapping[str, object]],
) -> np.ndarray:
    skeleton_mask = skeletonize(vessel_mask)
    if not np.any(skeleton_mask):
        return skeleton_mask

    segment_mask = np.zeros_like(skeleton_mask, dtype=bool)
    for measurement in measurements:
        start_yx = _nearest_skeleton_coordinate(
            skeleton_mask,
            x_value=float(measurement["x_start"]),
            y_value=float(measurement["y_start"]),
        )
        end_yx = _nearest_skeleton_coordinate(
            skeleton_mask,
            x_value=float(measurement["x_end"]),
            y_value=float(measurement["y_end"]),
        )
        if start_yx is None or end_yx is None:
            continue
        segment_mask |= _trace_skeleton_segment(skeleton_mask, start_yx, end_yx)
    return segment_mask


def create_fundus_overlay(
    rgb_path: str,
    av_path: Optional[str] = None,
    disc_path: Optional[str] = None,
    vessel_path: Optional[str] = None,
    circle_paths: Optional[Mapping[str, str]] = None,
    vessel_width_measurements: Optional[Sequence[Mapping[str, object]]] = None,
    tortuosity_measurements: Optional[Sequence[Mapping[str, object]]] = None,
    fovea_location: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    overlay_config: Optional[OverlayConfig] = None,
) -> np.ndarray:
    """
    Create a visualization of a fundus image with overlaid segmentations and markers.

    Args:
        rgb_path: Path to the RGB fundus image
        av_path: Optional path to artery-vein segmentation (1=artery, 2=vein, 3=intersection)
        disc_path: Optional path to binary disc segmentation
        vessel_path: Optional path to binary vessel segmentation for clipping measurement overlays
        circle_paths: Optional mapping from circle names to binary circle-mask paths
        vessel_width_measurements: Optional sequence of kept width-measurement records to draw
        tortuosity_measurements: Optional sequence of tortuosity records whose endpoint
            chord lines should be drawn; when provided with vessel_path, the vessel
            skeleton is also overlaid
        fovea_location: Optional (x,y) tuple indicating the location of the fovea
        output_path: Optional path to save the visualization image
        overlay_config: Overlay display configuration including enabled layers and colors

    Returns:
        Numpy array containing the visualization image
    """
    overlay_config = overlay_config or OverlayConfig()

    # Load RGB image
    rgb_img = np.array(Image.open(rgb_path))

    # Create output image starting with the RGB image
    output_img = rgb_img.copy()

    # Load and overlay AV segmentation if provided
    if av_path and (overlay_config.layers.arteries or overlay_config.layers.veins):
        av_mask = np.array(Image.open(av_path))

        # Create masks for arteries (1), veins (2) and intersections (3)
        artery_mask = av_mask == 1
        vein_mask = av_mask == 2
        intersection_mask = av_mask == 3

        if overlay_config.layers.arteries:
            artery_combined = np.logical_or(artery_mask, intersection_mask)
            output_img[artery_combined, :] = overlay_config.colors.artery

        if overlay_config.layers.veins:
            vein_combined = np.logical_or(vein_mask, intersection_mask)
            output_img[vein_combined, :] = overlay_config.colors.vein

    # Load and overlay optic disc segmentation if provided
    if disc_path and overlay_config.layers.disc:
        disc_mask = np.array(Image.open(disc_path)) > 0
        output_img[disc_mask, :] = overlay_config.colors.disc

    for circle in overlay_config.circles:
        circle_path = (
            circle_paths.get(circle.name) if circle_paths is not None else None
        )
        if circle_path:
            circle_mask = np.array(Image.open(circle_path)) > 0
            output_img[circle_mask, :] = circle.color

    vessel_mask = None
    if vessel_path is not None:
        vessel_mask = np.array(Image.open(vessel_path)) > 0

    if vessel_width_measurements and overlay_config.layers.vessel_widths:
        measurement_mask = _rasterize_vessel_width_measurements(
            image_shape=output_img.shape[:2],
            measurements=vessel_width_measurements,
            vessel_mask=vessel_mask,
        )
        output_img[measurement_mask, :] = overlay_config.colors.vessel_width

    # Convert to PIL image for drawing the fovea marker
    pil_img = Image.fromarray(output_img)
    draw = ImageDraw.Draw(pil_img)

    # Add fovea marker if provided
    if fovea_location and overlay_config.layers.fovea:
        x, y = fovea_location
        marker_size = (
            min(pil_img.width, pil_img.height) // 50
        )  # Scale marker with image

        # Draw yellow X at fovea location
        draw.line(
            [(x - marker_size, y - marker_size), (x + marker_size, y + marker_size)],
            fill=overlay_config.colors.fovea,
            width=2,
        )
        draw.line(
            [(x - marker_size, y + marker_size), (x + marker_size, y - marker_size)],
            fill=overlay_config.colors.fovea,
            width=2,
        )

    # Convert back to numpy array
    output_img = np.array(pil_img)

    if tortuosity_measurements:
        if vessel_mask is not None:
            vessel_skeleton_mask = _rasterize_tortuosity_skeleton_segments(
                vessel_mask,
                tortuosity_measurements,
            )
            output_img[vessel_skeleton_mask, :] = overlay_config.colors.vessel
        tortuosity_mask = _rasterize_line_segments(
            image_shape=output_img.shape[:2],
            measurements=tortuosity_measurements,
        )
        output_img[tortuosity_mask, :] = overlay_config.colors.vessel

    # Save output if path provided
    if output_path:
        Image.fromarray(output_img).save(output_path)

    return output_img


def batch_create_overlays(
    rgb_dir: Path,
    output_dir: Path,
    av_dir: Optional[Path] = None,
    disc_dir: Optional[Path] = None,
    vessels_dir: Optional[Path] = None,
    circle_dirs: Optional[Mapping[str, Path]] = None,
    vessel_width_data: Optional[pd.DataFrame] = None,
    tortuosity_data: Optional[pd.DataFrame] = None,
    fovea_data: Optional[Dict[str, Tuple[int, int]]] = None,
    overlay_config: Optional[OverlayConfig] = None,
) -> None:
    """
    Create visualization overlays for a batch of images.

    Args:
        rgb_dir: Directory containing RGB fundus images
        output_dir: Directory to save visualization images
        av_dir: Optional directory containing AV segmentations
        disc_dir: Optional directory containing disc segmentations
        vessels_dir: Optional directory containing vessel segmentations used to clip width overlays
        circle_dirs: Optional mapping from circle names to directories containing circle masks
        vessel_width_data: Optional dataframe containing kept width-measurement records
        tortuosity_data: Optional dataframe containing tortuosity segment records
        fovea_data: Optional dictionary mapping image IDs to fovea coordinates
        overlay_config: Overlay display configuration including enabled layers and colors

    Returns:
        List of paths to created visualization images
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    overlay_config = overlay_config or OverlayConfig()
    measurements_by_image: dict[str, list[dict[str, object]]] = {}
    if vessel_width_data is not None and not vessel_width_data.empty:
        for image_id, group in vessel_width_data.groupby("image_id"):
            measurements_by_image[str(image_id)] = group.to_dict(orient="records")
    tortuosities_by_image: dict[str, list[dict[str, object]]] = {}
    if tortuosity_data is not None and not tortuosity_data.empty:
        for image_id, group in tortuosity_data.groupby("image_id"):
            tortuosities_by_image[str(image_id)] = group.to_dict(orient="records")

    # Get all RGB images
    rgb_files = list(rgb_dir.glob("*.png"))
    if not rgb_files:
        logger.warning("No RGB images found for overlays in %s", rgb_dir)
        return []
    logger.info("Creating overlays for %d images", len(rgb_files))

    # Process each image
    for rgb_file in rgb_files:
        image_id = rgb_file.stem

        # Check for corresponding AV segmentation
        av_file = None
        if av_dir:
            av_file_path = av_dir / f"{image_id}.png"
            if av_file_path.exists():
                av_file = str(av_file_path)

        # Check for corresponding disc segmentation
        disc_file = None
        if disc_dir:
            disc_file_path = disc_dir / f"{image_id}.png"
            if disc_file_path.exists():
                disc_file = str(disc_file_path)

        vessel_file = None
        if vessels_dir:
            vessel_file_path = vessels_dir / f"{image_id}.png"
            if vessel_file_path.exists():
                vessel_file = str(vessel_file_path)

        circle_files: dict[str, str] = {}
        if circle_dirs:
            for circle in overlay_config.circles:
                circle_dir = circle_dirs.get(circle.name)
                if circle_dir is None:
                    continue
                circle_file_path = circle_dir / f"{image_id}.png"
                if circle_file_path.exists():
                    circle_files[circle.name] = str(circle_file_path)

        # Get fovea location if available
        fovea_location = None
        if fovea_data and image_id in fovea_data:
            fovea_location = fovea_data[image_id]

        # Create output path
        output_file = output_dir / f"{image_id}.png"

        # Create and save overlay
        create_fundus_overlay(
            rgb_path=str(rgb_file),
            av_path=av_file,
            disc_path=disc_file,
            vessel_path=vessel_file,
            circle_paths=circle_files,
            vessel_width_measurements=measurements_by_image.get(image_id),
            tortuosity_measurements=tortuosities_by_image.get(image_id),
            fovea_location=fovea_location,
            output_path=str(output_file),
            overlay_config=overlay_config,
        )
    logger.info("Finished overlay generation in %s", output_dir)
