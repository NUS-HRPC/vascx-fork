from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vascx_models.config import OverlayCircle
from vascx_models.vessel_widths import (
    compute_revised_crx_from_widths,
    measure_vessel_width_at_coordinate,
    measure_vessel_widths_between_disc_circle_pair,
    resolve_vessel_width_circle_pair,
    select_vessel_width_measurements_for_equivalents,
)


def _write_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


def test_measure_vessel_width_at_coordinate_uses_local_skeleton_tangent() -> None:
    height = width = 160
    vessel = np.zeros((height, width), dtype=bool)
    x_center = 80
    vessel[:, x_center - 3 : x_center + 4] = True

    width_px, start_xy, end_xy = measure_vessel_width_at_coordinate(
        vessel_mask=vessel,
        point_xy=np.array([80.0, 80.0], dtype=float),
    )

    assert width_px == 7.0
    assert sorted([start_xy[0], end_xy[0]]) == pytest.approx([76.5, 83.5])
    assert start_xy[1] == pytest.approx(80.0)
    assert end_xy[1] == pytest.approx(80.0)


def test_compute_revised_crx_from_widths_records_selected_vessel_ids() -> None:
    records = []
    for vessel_type, widths in {
        "artery": [10.0, 14.0, 8.0],
        "vein": [20.0, 18.0],
    }.items():
        for connection_index, width in enumerate(widths, start=1):
            for sample_index in range(1, 3):
                records.append(
                    {
                        "image_id": "sample",
                        "inner_circle": "2r",
                        "outer_circle": "3r",
                        "inner_circle_radius_px": 40.0,
                        "outer_circle_radius_px": 60.0,
                        "connection_index": connection_index,
                        "sample_index": sample_index,
                        "x": float(connection_index),
                        "y": float(sample_index),
                        "width_px": width,
                        "x_start": 0.0,
                        "y_start": 0.0,
                        "x_end": 1.0,
                        "y_end": 1.0,
                        "vessel_type": vessel_type,
                    }
                )
    df_widths = pd.DataFrame.from_records(records)

    df_connections, df_equivalents = compute_revised_crx_from_widths(df_widths)
    selected = select_vessel_width_measurements_for_equivalents(
        df_widths,
        df_connections,
    )

    assert df_equivalents["metric"].tolist() == ["CRAE", "CRVE"]
    assert df_equivalents["requested_n_largest"].tolist() == [6, 6]
    assert df_equivalents["n_vessels_available"].tolist() == [3, 2]
    assert df_equivalents["n_vessels_used"].tolist() == [3, 2]
    assert df_equivalents["vessel_ids_used"].tolist() == [
        "artery_2;artery_1;artery_3",
        "vein_1;vein_2",
    ]
    assert np.isfinite(df_equivalents["equivalent_px"]).all()
    assert set(selected["vessel_type"]) == {"artery", "vein"}
    assert len(selected) == len(df_widths)


def test_resolve_vessel_width_circle_pair_uses_named_circles_when_provided() -> None:
    circles = (
        OverlayCircle(name="3r", diameter=3.0),
        OverlayCircle(name="2r", diameter=2.0),
        OverlayCircle(name="5r", diameter=5.0),
    )

    inner_circle, outer_circle = resolve_vessel_width_circle_pair(
        circles,
        inner_circle_name="2r",
        outer_circle_name="5r",
    )

    assert inner_circle.name == "2r"
    assert outer_circle.name == "5r"


def test_resolve_vessel_width_circle_pair_defaults_to_two_smallest_valid_circles() -> None:
    circles = (
        OverlayCircle(name="5r", diameter=5.0),
        OverlayCircle(name="2r", diameter=2.0),
        OverlayCircle(name="3r", diameter=3.0),
    )

    inner_circle, outer_circle = resolve_vessel_width_circle_pair(circles)

    assert inner_circle.name == "2r"
    assert outer_circle.name == "3r"


def test_measure_vessel_widths_between_disc_circle_pair_writes_empty_csv_when_no_connections(
    tmp_path: Path,
) -> None:
    vessels_dir = tmp_path / "vessels"
    av_dir = tmp_path / "artery_vein"
    vessels_dir.mkdir()
    av_dir.mkdir()

    empty = np.zeros((64, 64), dtype=np.uint8)
    _write_mask(vessels_dir / "sample.png", empty)
    _write_mask(av_dir / "sample.png", empty)

    geometry_path = tmp_path / "disc_geometry.csv"
    pd.DataFrame(
        {
            "x_disc_center": [32.0],
            "y_disc_center": [32.0],
            "disc_radius_px": [10.0],
        },
        index=["sample"],
    ).to_csv(geometry_path)

    output_path = tmp_path / "vessel_widths.csv"
    df = measure_vessel_widths_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
        output_path=output_path,
    )

    assert output_path.exists()
    assert df.empty
    assert list(df.columns) == [
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
    ]


def test_measure_vessel_widths_between_disc_circle_pair_samples_interior_points(tmp_path: Path) -> None:
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

    df = measure_vessel_widths_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
        samples_per_connection=5,
    )

    assert len(df) == 10
    assert list(df["inner_circle"].unique()) == ["inner"]
    assert list(df["outer_circle"].unique()) == ["outer"]
    assert sorted(df["connection_index"].unique().tolist()) == [1, 2]
    assert df.groupby("connection_index")["sample_index"].apply(list).tolist() == [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ]
    assert df["width_px"].tolist() == [7.0] * 10
    assert df["vessel_type"].tolist() == ["vein"] * 10
    assert df["x"].tolist() == [80.0] * 10
    assert not df["y"].isin([20.0, 40.0, 120.0, 140.0]).any()
    assert sorted(df["y"].tolist()) == pytest.approx([
        23.333333333333332,
        26.666666666666664,
        30.0,
        33.333333333333336,
        36.666666666666664,
        123.33333333333333,
        126.66666666666667,
        130.0,
        133.33333333333334,
        136.66666666666666,
    ])


def test_measure_vessel_widths_between_disc_circle_pair_separates_arteries_and_veins(
    tmp_path: Path,
) -> None:
    vessels_dir = tmp_path / "vessels"
    av_dir = tmp_path / "artery_vein"
    vessels_dir.mkdir()
    av_dir.mkdir()

    height = width = 200
    vessel = np.zeros((height, width), dtype=np.uint8)
    av = np.zeros((height, width), dtype=np.uint8)

    vessel[:, 70:73] = 1
    av[:, 70:73] = 1

    vessel[:, 127:134] = 1
    av[:, 127:134] = 2

    _write_mask(vessels_dir / "sample.png", vessel)
    _write_mask(av_dir / "sample.png", av)

    geometry_path = tmp_path / "disc_geometry.csv"
    pd.DataFrame(
        {
            "x_disc_center": [100.0],
            "y_disc_center": [100.0],
            "disc_radius_px": [20.0],
        },
        index=["sample"],
    ).to_csv(geometry_path)

    df = measure_vessel_widths_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
        samples_per_connection=5,
    )

    assert len(df) == 20
    assert sorted(df["vessel_type"].unique().tolist()) == ["artery", "vein"]
    assert df.groupby("vessel_type").size().to_dict() == {"artery": 10, "vein": 10}
    assert sorted(df.groupby("vessel_type")["width_px"].first().tolist()) == [3.0, 7.0]


def test_measure_vessel_widths_between_disc_circle_pair_prunes_dead_end_branch(
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
    vessel[:, x_center] = 1
    av[:, x_center] = 2

    # Add a one-pixel T-branch inside the annulus that does not reach the outer circle.
    vessel[125, x_center:91] = 1
    av[125, x_center:91] = 2

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

    df = measure_vessel_widths_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
        samples_per_connection=5,
    )

    assert len(df) == 10
    assert sorted(df["connection_index"].unique().tolist()) == [1, 2]
    assert df["x"].tolist() == [80.0] * 10
    assert sorted(df["y"].tolist()) == pytest.approx([
        23.333333333333332,
        26.666666666666664,
        30.0,
        33.333333333333336,
        36.666666666666664,
        123.33333333333333,
        126.66666666666667,
        130.0,
        133.33333333333334,
        136.66666666666666,
    ])


def test_measure_vessel_widths_between_disc_circle_pair_measures_one_to_many_fork(
    tmp_path: Path,
) -> None:
    vessels_dir = tmp_path / "vessels"
    av_dir = tmp_path / "artery_vein"
    vessels_dir.mkdir()
    av_dir.mkdir()

    height = width = 180
    vessel = np.zeros((height, width), dtype=np.uint8)
    av = np.zeros((height, width), dtype=np.uint8)

    # One inner-circle trunk contact forks into two daughter branches that
    # both reach the outer circle.
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

    df = measure_vessel_widths_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
        samples_per_connection=5,
    )

    assert len(df) == 5
    assert sorted(df["connection_index"].unique().tolist()) == [1]
    assert (df["width_px"] > 0).all()

    assert df["x"].tolist() == [90.0] * 5
    assert df["y"].tolist() == pytest.approx([
        130.83333333333334,
        131.66666666666666,
        132.5,
        133.33333333333334,
        134.16666666666666,
    ])


def test_measure_vessel_widths_between_disc_circle_pair_measures_many_to_many_fork(
    tmp_path: Path,
) -> None:
    vessels_dir = tmp_path / "vessels"
    av_dir = tmp_path / "artery_vein"
    vessels_dir.mkdir()
    av_dir.mkdir()

    height = width = 180
    vessel = np.zeros((height, width), dtype=np.uint8)
    av = np.zeros((height, width), dtype=np.uint8)

    # Two inner-circle trunks join a shared branch that reaches two outer-circle contacts.
    vessel[130:136, 80] = 1
    av[130:136, 80] = 2
    vessel[130:136, 100] = 1
    av[130:136, 100] = 2
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

    df = measure_vessel_widths_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
        samples_per_connection=5,
    )

    assert len(df) == 10
    assert sorted(df["connection_index"].unique().tolist()) == [1, 2]
    assert (df["width_px"] > 0).all()

    by_connection = {
        connection_index: group
        for connection_index, group in df.groupby("connection_index")
    }
    assert by_connection[1]["x"].tolist() == [80.0] * 5
    assert by_connection[2]["x"].tolist() == [100.0] * 5
    assert by_connection[1]["y"].tolist() == pytest.approx([
        130.83333333333334,
        131.66666666666666,
        132.5,
        133.33333333333334,
        134.16666666666666,
    ])
    assert by_connection[2]["y"].tolist() == pytest.approx([
        130.83333333333334,
        131.66666666666666,
        132.5,
        133.33333333333334,
        134.16666666666666,
    ])
