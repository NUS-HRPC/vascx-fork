from pathlib import Path

import numpy as np
from PIL import Image

from vascx_models.config import OverlayCircle
from vascx_models.geometry.disc_circles import generate_disc_circles


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8)).save(path)


def test_generate_disc_circles_writes_configured_circle_masks_and_measurements(
    tmp_path: Path,
) -> None:
    disc_dir = tmp_path / "disc"
    circle_output_dir = tmp_path / "disc_circles"
    disc_dir.mkdir()

    mask = np.zeros((80, 80), dtype=np.uint8)
    yy, xx = np.indices(mask.shape)
    circle = (xx - 40) ** 2 + (yy - 40) ** 2 <= 8**2
    mask[circle] = 1
    _save_mask(disc_dir / "sample.png", mask)

    circles = (
        OverlayCircle(name="inner", diameter=1.5),
        OverlayCircle(name="outer", diameter=2.5),
    )
    df = generate_disc_circles(
        disc_dir=disc_dir,
        circle_output_dir=circle_output_dir,
        circles=circles,
    )

    assert list(df.columns) == [
        "x_disc_center",
        "y_disc_center",
        "disc_radius_px",
        "circle_inner_px",
        "circle_outer_px",
    ]
    assert df.index.tolist() == ["sample"]
    assert df.loc["sample", "circle_inner_px"] > 0
    assert df.loc["sample", "circle_outer_px"] > df.loc["sample", "circle_inner_px"]

    inner_mask = np.array(Image.open(circle_output_dir / "inner" / "sample.png"))
    outer_mask = np.array(Image.open(circle_output_dir / "outer" / "sample.png"))
    assert np.any(inner_mask > 0)
    assert np.any(outer_mask > 0)


def test_generate_disc_circles_handles_empty_disc_masks(tmp_path: Path) -> None:
    disc_dir = tmp_path / "disc"
    circle_output_dir = tmp_path / "disc_circles"
    disc_dir.mkdir()

    empty_mask = np.zeros((32, 32), dtype=np.uint8)
    _save_mask(disc_dir / "empty.png", empty_mask)

    circles = (OverlayCircle(name="outer", diameter=2.0),)
    df = generate_disc_circles(
        disc_dir=disc_dir,
        circle_output_dir=circle_output_dir,
        circles=circles,
    )

    assert np.isnan(df.loc["empty", "disc_radius_px"])
    assert np.isnan(df.loc["empty", "circle_outer_px"])
    saved_circle = np.array(Image.open(circle_output_dir / "outer" / "empty.png"))
    assert not np.any(saved_circle)
