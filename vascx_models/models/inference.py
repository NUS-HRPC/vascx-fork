import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from rtnls_inference.ensembles.ensemble_classification import ClassificationEnsemble
from rtnls_inference.ensembles.ensemble_heatmap_regression import (
    HeatmapRegressionEnsemble,
)
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from rtnls_inference.utils import decollate_batch, extract_keypoints_from_heatmaps

from .model_assets import ensure_model_files_present

logger = logging.getLogger(__name__)


def _save_visual_mask(mask: np.ndarray, path: str, color_by_value: dict[int, tuple[int, int, int]]) -> None:
    """Save a label mask with a palette while preserving label values."""
    mask_uint8 = mask.squeeze().astype(np.uint8)
    image = Image.fromarray(mask_uint8)
    palette = [0] * (256 * 3)
    for value, color in color_by_value.items():
        start = int(value) * 3
        palette[start : start + 3] = list(color)
    image.putpalette(palette)
    image.save(path)


def available_device_types() -> dict[str, bool]:
    return {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }


def preferred_device() -> torch.device:
    return resolve_device("auto")


def resolve_device(device_name: str = "auto") -> torch.device:
    available = available_device_types()
    if device_name == "auto":
        if available["cuda"]:
            return torch.device("cuda:0")
        if available["mps"]:
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda":
        if not available["cuda"]:
            raise RuntimeError("Requested device 'cuda' is not available")
        return torch.device("cuda:0")
    if device_name == "mps":
        if not available["mps"]:
            raise RuntimeError("Requested device 'mps' is not available")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{device_name}'")


def _inference_num_workers(device: torch.device) -> int:
    # Torch shared-memory workers can fail in restricted CPU environments.
    return 8 if device.type in {"cuda", "mps"} else 0


def _autocast_context(device: torch.device):
    return torch.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()


def run_quality_estimation(fpaths, ids, device: torch.device):
    ensure_model_files_present(["quality.pt"])
    logger.info("Loading quality model on %s", device)
    ensemble_quality = ClassificationEnsemble.from_release("quality.pt").to(device)
    dataloader = ensemble_quality._make_inference_dataloader(
        fpaths,
        ids=ids,
        num_workers=_inference_num_workers(device),
        preprocess=False,
        batch_size=16,
    )
    logger.info("Quality dataloader ready with %d images", len(fpaths))

    output_ids, outputs = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            im = batch["image"].to(device)

            # QUALITY
            quality = ensemble_quality.predict_step(im)
            quality = torch.mean(quality, dim=0)

            items = {"id": batch["id"], "quality": quality}
            items = decollate_batch(items)

            for item in items:
                output_ids.append(item["id"])
                outputs.append(item["quality"].tolist())

    return pd.DataFrame(
        outputs,
        index=output_ids,
        columns=["q1", "q2", "q3"],
    )


def run_segmentation_vessels_and_av(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
    av_path: Optional[Path] = None,
    vessels_path: Optional[Path] = None,
    artery_color: tuple[int, int, int] = (255, 0, 0),
    vein_color: tuple[int, int, int] = (0, 0, 255),
    vessel_color: tuple[int, int, int] = (255, 255, 255),
    device: torch.device = preferred_device(),
) -> None:
    """
    Run AV and vessel segmentation on the provided images.

    Args:
        rgb_paths: List of paths to RGB fundus images
        ce_paths: Optional list of paths to contrast enhanced images
        ids: Optional list of ids to pass to _make_inference_dataloader
        av_path: Folder where to store output AV segmentations
        vessels_path: Folder where to store output vessel segmentations
        artery_color: Display color for artery class in saved AV masks
        vein_color: Display color for vein class in saved AV masks
        vessel_color: Display color for vessel class in saved vessel masks
        device: Device to run inference on
    """
    # Create output directories if they don't exist
    if av_path is not None:
        av_path.mkdir(exist_ok=True, parents=True)
    if vessels_path is not None:
        vessels_path.mkdir(exist_ok=True, parents=True)

    # Load models
    ensure_model_files_present(["av_july24.pt", "vessels_july24.pt"])
    logger.info("Loading AV and vessel models on %s", device)
    ensemble_av = SegmentationEnsemble.from_release("av_july24.pt").to(device).eval()
    ensemble_vessels = (
        SegmentationEnsemble.from_release("vessels_july24.pt").to(device).eval()
    )

    # Prepare input paths
    if ce_paths is None:
        # If CE paths are not provided, use RGB paths for both inputs
        fpaths = rgb_paths
    else:
        # If CE paths are provided, pair them with RGB paths
        if len(rgb_paths) != len(ce_paths):
            raise ValueError("rgb_paths and ce_paths must have the same length")
        fpaths = list(zip(rgb_paths, ce_paths))

    # Create dataloader
    dataloader = ensemble_av._make_inference_dataloader(
        fpaths,
        ids=ids,
        num_workers=_inference_num_workers(device),
        preprocess=False,
        batch_size=8,
    )
    logger.info("AV and vessel dataloader ready with %d images", len(fpaths))

    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # AV segmentation
            if av_path is not None:
                with _autocast_context(device):
                    proba = ensemble_av.forward(batch["image"].to(device))
                proba = torch.mean(proba, dim=1)  # average over models
                proba = torch.permute(proba, (0, 2, 3, 1))  # NCHW -> NHWC
                proba = torch.nn.functional.softmax(proba, dim=-1)

                items = {
                    "id": batch["id"],
                    "image": proba,
                }

                items = decollate_batch(items)
                for i, item in enumerate(items):
                    fpath = os.path.join(av_path, f"{item['id']}.png")
                    mask = np.argmax(item["image"], -1)
                    _save_visual_mask(
                        mask,
                        fpath,
                        {
                            1: artery_color,
                            2: vein_color,
                            3: (255, 255, 255),
                        },
                    )

            # Vessel segmentation
            if vessels_path is not None:
                with _autocast_context(device):
                    proba = ensemble_vessels.forward(batch["image"].to(device))
                proba = torch.mean(proba, dim=1)  # average over models
                proba = torch.permute(proba, (0, 2, 3, 1))  # NCHW -> NHWC
                proba = torch.nn.functional.softmax(proba, dim=-1)

                items = {
                    "id": batch["id"],
                    "image": proba,
                }

                items = decollate_batch(items)
                for i, item in enumerate(items):
                    fpath = os.path.join(vessels_path, f"{item['id']}.png")
                    mask = np.argmax(item["image"], -1)
                    _save_visual_mask(mask, fpath, {1: vessel_color})


def run_segmentation_disc(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    disc_color: tuple[int, int, int] = (255, 255, 255),
    device: torch.device = preferred_device(),
) -> None:
    ensure_model_files_present(["disc_july24.pt"])
    logger.info("Loading disc model on %s", device)
    ensemble_disc = (
        SegmentationEnsemble.from_release("disc_july24.pt").to(device).eval()
    )

    # Prepare input paths
    if ce_paths is None:
        # If CE paths are not provided, use RGB paths for both inputs
        fpaths = rgb_paths
    else:
        # If CE paths are provided, pair them with RGB paths
        if len(rgb_paths) != len(ce_paths):
            raise ValueError("rgb_paths and ce_paths must have the same length")
        fpaths = list(zip(rgb_paths, ce_paths))

    dataloader = ensemble_disc._make_inference_dataloader(
        fpaths,
        ids=ids,
        num_workers=_inference_num_workers(device),
        preprocess=False,
        batch_size=8,
    )
    logger.info("Disc dataloader ready with %d images", len(fpaths))

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # AV
            with _autocast_context(device):
                proba = ensemble_disc.forward(batch["image"].to(device))
            proba = torch.mean(proba, dim=1)  # average over models
            proba = torch.permute(proba, (0, 2, 3, 1))  # NCHW -> NHWC
            proba = torch.nn.functional.softmax(proba, dim=-1)

            items = {
                "id": batch["id"],
                "image": proba,
            }

            items = decollate_batch(items)
            items = [dataloader.dataset.transform.undo_item(item) for item in items]
            for i, item in enumerate(items):
                fpath = os.path.join(output_path, f"{item['id']}.png")

                mask = np.argmax(item["image"], -1)
                _save_visual_mask(mask, fpath, {1: disc_color})


def run_fovea_detection(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
    device: torch.device = preferred_device(),
) -> None:
    # def run_fovea_detection(fpaths, ids, device: torch.device):
    ensure_model_files_present(["fovea_july24.pt"])
    logger.info("Loading fovea model on %s", device)
    ensemble_fovea = HeatmapRegressionEnsemble.from_release("fovea_july24.pt").to(
        device
    )

    # Prepare input paths
    if ce_paths is None:
        # If CE paths are not provided, use RGB paths for both inputs
        fpaths = rgb_paths
    else:
        # If CE paths are provided, pair them with RGB paths
        if len(rgb_paths) != len(ce_paths):
            raise ValueError("rgb_paths and ce_paths must have the same length")
        fpaths = list(zip(rgb_paths, ce_paths))

    dataloader = ensemble_fovea._make_inference_dataloader(
        fpaths,
        ids=ids,
        num_workers=_inference_num_workers(device),
        preprocess=False,
        batch_size=8,
    )
    logger.info("Fovea dataloader ready with %d images", len(fpaths))

    output_ids, outputs = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            im = batch["image"].to(device)

            # FOVEA DETECTION
            with _autocast_context(device):
                heatmap = ensemble_fovea.forward(im)
            keypoints = extract_keypoints_from_heatmaps(heatmap)

            kp_fovea = torch.mean(keypoints, dim=1)  # average over models

            items = {
                "id": batch["id"],
                "keypoints": kp_fovea,
                "metadata": batch["metadata"],
            }
            items = decollate_batch(items)

            items = [dataloader.dataset.transform.undo_item(item) for item in items]

            for item in items:
                output_ids.append(item["id"])
                outputs.append(
                    [
                        *item["keypoints"][0].tolist(),
                    ]
                )
    return pd.DataFrame(
        outputs,
        index=output_ids,
        columns=["x_fovea", "y_fovea"],
    )
