from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vascx_models.config import OverlayCircle
from vascx_models.vessel_tortuosities import (
    compute_path_tortuosity,
    measure_vessel_tortuosities_between_disc_circle_pair,
    vessel_tortuosity_record,
)


def _write_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


def test_compute_path_tortuosity_uses_path_length_over_chord() -> None:
    path_xy = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 4.0],
        ],
        dtype=float,
    )

    path_length_px, chord_length_px, tortuosity = compute_path_tortuosity(path_xy)

    assert path_length_px == pytest.approx(8.0)
    assert chord_length_px == pytest.approx(np.hypot(6.0, 4.0))
    assert tortuosity == pytest.approx(8.0 / np.hypot(6.0, 4.0))


def test_vessel_tortuosity_record_uses_ordered_path_endpoints() -> None:
    inner_circle = OverlayCircle(name="2r", diameter=2.0)
    outer_circle = OverlayCircle(name="3r", diameter=3.0)
    path_xy = np.array([[10.0, 20.0], [13.0, 24.0], [16.0, 24.0]], dtype=float)

    record = vessel_tortuosity_record(
        image_id="sample",
        vessel_type="artery",
        inner_circle=inner_circle,
        outer_circle=outer_circle,
        inner_radius_px=40.0,
        outer_radius_px=60.0,
        connection_index=7,
        path_xy=path_xy,
    )

    assert record["image_id"] == "sample"
    assert record["vessel_type"] == "artery"
    assert record["connection_index"] == 7
    assert record["inner_circle"] == "2r"
    assert record["outer_circle"] == "3r"
    assert record["x_start"] == 10.0
    assert record["y_start"] == 20.0
    assert record["x_end"] == 16.0
    assert record["y_end"] == 24.0
    assert record["tortuosity"] == pytest.approx(8.0 / np.hypot(6.0, 4.0))


def test_measure_vessel_tortuosities_between_disc_circle_pair_writes_vessel_tortuosity(
    tmp_path: Path,
) -> None:
    vessels_dir = tmp_path / "vessels"
    av_dir = tmp_path / "artery_vein"
    vessels_dir.mkdir()
    av_dir.mkdir()

    height = width = 160
    vessel = np.zeros((height, width), dtype=np.uint8)
    av = np.zeros((height, width), dtype=np.uint8)

    x_center = 80
    vessel[:, x_center - 3 : x_center + 4] = 1
    av[:, x_center - 3 : x_center + 4] = 2

    _write_mask(vessels_dir / "sample.png", vessel)
    _write_mask(av_dir / "sample.png", av)

    geometry_path = tmp_path / "disc_geometry.csv"
    pd.DataFrame(
        {
            "x_disc_center": [80.0],
            "y_disc_center": [80.0],
            "disc_radius_px": [20.0],
        },
        index=["sample"],
    ).to_csv(geometry_path)

    tortuosity_output_path = tmp_path / "vessel_tortuosities.csv"
    df_tortuosities = measure_vessel_tortuosities_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
        output_path=tortuosity_output_path,
    )

    assert tortuosity_output_path.exists()
    assert df_tortuosities.columns.tolist() == [
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
    assert len(df_tortuosities) == 2
    assert df_tortuosities["tortuosity"].tolist() == pytest.approx([1.0, 1.0])
    assert (df_tortuosities["path_length_px"] > 0).all()
    assert df_tortuosities["path_length_px"].tolist() == pytest.approx(
        df_tortuosities["chord_length_px"].tolist()
    )


def test_measure_vessel_tortuosities_between_disc_circle_pair_measures_one_to_many_fork(
    tmp_path: Path,
) -> None:
    vessels_dir = tmp_path / "vessels"
    av_dir = tmp_path / "artery_vein"
    vessels_dir.mkdir()
    av_dir.mkdir()

    height = width = 180
    vessel = np.zeros((height, width), dtype=np.uint8)
    av = np.zeros((height, width), dtype=np.uint8)

    vessel[130:136, 90] = 1
    av[130:136, 90] = 2
    vessel[135, 40:141] = 1
    av[135, 40:141] = 2

    _write_mask(vessels_dir / "sample.png", vessel)
    _write_mask(av_dir / "sample.png", av)

    geometry_path = tmp_path / "disc_geometry.csv"
    pd.DataFrame(
        {
            "x_disc_center": [90.0],
            "y_disc_center": [90.0],
            "disc_radius_px": [20.0],
        },
        index=["sample"],
    ).to_csv(geometry_path)

    df_tortuosities = measure_vessel_tortuosities_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
    )

    assert len(df_tortuosities) == 2
    assert sorted(df_tortuosities["connection_index"].tolist()) == [1, 2]
    assert df_tortuosities["vessel_type"].tolist() == ["vein", "vein"]
    assert sorted(df_tortuosities["x_end"].tolist()) == [51.0, 129.0]
    assert df_tortuosities["path_length_px"].tolist() == pytest.approx([44.0, 44.0])
