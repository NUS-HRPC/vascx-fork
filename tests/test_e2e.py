import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from rtnls_fundusprep.cli import _run_preprocessing

from vascx_models.config import AppConfig
from vascx_models.geometry.disc_circles import generate_disc_circles
from vascx_models.metrics.vessel_widths import (
    VESSEL_WIDTH_COLUMNS,
    measure_vessel_widths_between_disc_circle_pair,
    resolve_vessel_width_circle_pair,
)
from vascx_models.models.inference import (
    run_fovea_detection,
    run_quality_estimation,
    run_segmentation_disc,
    run_segmentation_vessels_and_av,
)
from vascx_models.models.model_assets import missing_model_paths
from vascx_models.overlays.utils import batch_create_overlays
from vascx_models.runtime import configure_runtime_environment

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "samples" / "fundus" / "original" / "DRIVE_22.png"
EXPECTED_VESSEL_WIDTH_COLUMNS = VESSEL_WIDTH_COLUMNS


def _require_e2e_opt_in() -> None:
    if os.environ.get("VASCX_RUN_E2E") != "1":
        pytest.skip("Set VASCX_RUN_E2E=1 to run real-model end-to-end tests")
    if missing_model_paths(REPO_ROOT):
        pytest.skip("Run ./setup.sh to download the required model weights first")


def _device_or_skip(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available in this environment")
        return torch.device("cuda:0")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("MPS is not available in this environment")
        return torch.device("mps")
    raise AssertionError(f"Unsupported device name: {device_name}")


def _prepare_single_image_input(tmp_path: Path) -> tuple[str, Path, Path]:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    image_path = input_dir / SAMPLE_IMAGE.name
    shutil.copy2(SAMPLE_IMAGE, image_path)
    return SAMPLE_IMAGE.stem, image_path, input_dir


def _assert_nonempty_mask(path: Path) -> None:
    assert path.exists()
    assert np.any(np.array(Image.open(path)) > 0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda", "mps"])
def test_single_image_pipeline_smoke(tmp_path: Path, device_name: str) -> None:
    _require_e2e_opt_in()
    configure_runtime_environment()
    device = _device_or_skip(device_name)
    app_config = AppConfig()

    image_id, image_path, input_dir = _prepare_single_image_input(tmp_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    preprocessed_rgb_dir = output_dir / "preprocessed_rgb"
    av_dir = output_dir / "artery_vein"
    vessels_dir = output_dir / "vessels"
    disc_dir = output_dir / "disc"
    disc_circles_dir = output_dir / "disc_circles"
    overlay_dir = output_dir / "overlays"
    preprocessed_rgb_dir.mkdir()
    av_dir.mkdir()
    vessels_dir.mkdir()
    disc_dir.mkdir()
    overlay_dir.mkdir()

    bounds_path = output_dir / "bounds.csv"
    quality_path = output_dir / "quality.csv"
    fovea_path = output_dir / "fovea.csv"
    disc_geometry_path = output_dir / "disc_geometry.csv"
    vessel_widths_path = output_dir / "vessel_widths.csv"

    _run_preprocessing(
        files=[image_path],
        ids=[image_id],
        rgb_path=preprocessed_rgb_dir,
        bounds_path=bounds_path,
        n_jobs=1,
    )
    preprocessed_image_path = preprocessed_rgb_dir / f"{image_id}.png"

    df_quality = run_quality_estimation(
        [preprocessed_image_path], ids=[image_id], device=device
    )
    df_quality.to_csv(quality_path)

    run_segmentation_vessels_and_av(
        rgb_paths=[preprocessed_image_path],
        ids=[image_id],
        av_path=av_dir,
        vessels_path=vessels_dir,
        artery_color=app_config.overlay.colors.artery,
        vein_color=app_config.overlay.colors.vein,
        vessel_color=app_config.overlay.colors.vessel,
        device=device,
    )
    run_segmentation_disc(
        rgb_paths=[preprocessed_image_path],
        ids=[image_id],
        output_path=disc_dir,
        disc_color=app_config.overlay.colors.disc,
        device=device,
    )

    df_disc_geometry = generate_disc_circles(
        disc_dir=disc_dir,
        circle_output_dir=disc_circles_dir,
        circles=app_config.overlay.circles,
        measurements_path=disc_geometry_path,
    )
    inner_circle, outer_circle = resolve_vessel_width_circle_pair(
        app_config.overlay.circles,
        inner_circle_name=app_config.vessel_widths.inner_circle,
        outer_circle_name=app_config.vessel_widths.outer_circle,
    )
    df_vessel_widths = measure_vessel_widths_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=disc_geometry_path,
        inner_circle=inner_circle,
        outer_circle=outer_circle,
        output_path=vessel_widths_path,
        samples_per_connection=app_config.vessel_widths.samples_per_connection,
    )

    df_fovea = run_fovea_detection(
        [preprocessed_image_path], ids=[image_id], device=device
    )
    df_fovea.to_csv(fovea_path)

    batch_create_overlays(
        rgb_dir=preprocessed_rgb_dir,
        output_dir=overlay_dir,
        av_dir=av_dir,
        disc_dir=disc_dir,
        vessels_dir=vessels_dir,
        circle_dirs={
            circle.name: disc_circles_dir / circle.name
            for circle in app_config.overlay.circles
        },
        vessel_width_data=df_vessel_widths,
        fovea_data={
            index: (row["x_fovea"], row["y_fovea"])
            for index, row in df_fovea.iterrows()
        },
        overlay_config=app_config.overlay,
    )

    assert df_quality.index.tolist() == [image_id]
    assert df_quality.columns.tolist() == ["q1", "q2", "q3"]
    assert np.isfinite(df_quality.to_numpy()).all()
    assert quality_path.exists()
    assert bounds_path.exists()
    assert preprocessed_image_path.exists()

    _assert_nonempty_mask(av_dir / f"{image_id}.png")
    _assert_nonempty_mask(vessels_dir / f"{image_id}.png")
    _assert_nonempty_mask(disc_dir / f"{image_id}.png")

    assert df_disc_geometry.index.tolist() == [image_id]
    assert float(df_disc_geometry.loc[image_id, "disc_radius_px"]) > 0.0
    assert disc_geometry_path.exists()
    for circle in app_config.overlay.circles:
        _assert_nonempty_mask(disc_circles_dir / circle.name / f"{image_id}.png")

    assert vessel_widths_path.exists()
    df_vessel_widths_disk = pd.read_csv(vessel_widths_path)
    assert df_vessel_widths_disk.columns.tolist() == EXPECTED_VESSEL_WIDTH_COLUMNS
    assert df_vessel_widths.columns.tolist() == EXPECTED_VESSEL_WIDTH_COLUMNS
    if not df_vessel_widths.empty:
        assert df_vessel_widths["image_id"].eq(image_id).all()
        assert df_vessel_widths["vessel_type"].isin(["artery", "vein"]).all()
        assert (df_vessel_widths["width_px"] > 0).all()

    assert df_fovea.index.tolist() == [image_id]
    assert df_fovea.columns.tolist() == ["x_fovea", "y_fovea"]
    assert np.isfinite(df_fovea.to_numpy()).all()
    assert fovea_path.exists()

    overlay_path = overlay_dir / f"{image_id}.png"
    assert overlay_path.exists()
    assert Image.open(overlay_path).size == Image.open(preprocessed_image_path).size
