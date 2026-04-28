from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vascx_models.config import OverlayCircle, ProfileWidthConfig, VesselWidthConfig
from vascx_models.metrics.vessel_widths import (
    VESSEL_WIDTH_COLUMNS,
    compute_revised_crx_from_widths,
    measure_vessel_width_at_coordinate,
    measure_vessel_widths_between_disc_circle_pair,
    resolve_vessel_width_circle_pair,
)


def _write_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


def _write_rgb(path: Path, array: np.ndarray) -> None:
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


def test_resolve_vessel_width_circle_pair_defaults_to_two_smallest_valid_circles() -> (
    None
):
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
    assert list(df.columns) == VESSEL_WIDTH_COLUMNS


def test_measure_vessel_widths_between_disc_circle_pair_samples_interior_points(
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
    assert df["width_method"].tolist() == ["mask"] * 10
    assert df["measurement_valid"].tolist() == [True] * 10
    assert df["x"].tolist() == [80.0] * 10
    assert not df["y"].isin([20.0, 40.0, 120.0, 140.0]).any()
    assert sorted(df["y"].tolist()) == pytest.approx(
        [
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
        ]
    )


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
    assert df["width_method"].tolist() == ["mask"] * 20


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
    assert sorted(df["y"].tolist()) == pytest.approx(
        [
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
        ]
    )


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
    assert df["y"].tolist() == pytest.approx(
        [
            130.83333333333334,
            131.66666666666666,
            132.5,
            133.33333333333334,
            134.16666666666666,
        ]
    )


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
    assert by_connection[1]["y"].tolist() == pytest.approx(
        [
            130.83333333333334,
            131.66666666666666,
            132.5,
            133.33333333333334,
            134.16666666666666,
        ]
    )
    assert by_connection[2]["y"].tolist() == pytest.approx(
        [
            130.83333333333334,
            131.66666666666666,
            132.5,
            133.33333333333334,
            134.16666666666666,
        ]
    )


def test_measure_vessel_widths_between_disc_circle_pair_negative_samples_use_every_interior_path_pixel(
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

    df = measure_vessel_widths_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
        samples_per_connection=-1,
    )

    assert len(df) == 38
    assert sorted(df["connection_index"].unique().tolist()) == [1, 2]
    assert df.groupby("connection_index")["sample_index"].apply(list).tolist() == [
        list(range(1, 20)),
        list(range(1, 20)),
    ]
    assert sorted(df["y"].tolist()) == pytest.approx(
        [float(value) for value in range(21, 40)]
        + [float(value) for value in range(121, 140)]
    )


@pytest.mark.parametrize(
    ("width_config", "expected_method", "needs_rgb"),
    [
        (VesselWidthConfig(method="mask"), "mask", False),
        (VesselWidthConfig(method="pvbm_mask"), "pvbm_mask", False),
        (
            VesselWidthConfig(
                method="profile",
                profile=ProfileWidthConfig(image_source="preprocessed_rgb"),
            ),
            "profile",
            True,
        ),
    ],
)
def test_measure_vessel_widths_all_methods_preserve_equivalent_aggregation(
    tmp_path: Path,
    width_config: VesselWidthConfig,
    expected_method: str,
    needs_rgb: bool,
) -> None:
    vessels_dir = tmp_path / "vessels"
    av_dir = tmp_path / "artery_vein"
    rgb_dir = tmp_path / "preprocessed_rgb"
    vessels_dir.mkdir()
    av_dir.mkdir()
    if needs_rgb:
        rgb_dir.mkdir()

    height = width = 160
    vessel = np.zeros((height, width), dtype=np.uint8)
    av = np.zeros((height, width), dtype=np.uint8)
    rgb = np.full((height, width, 3), 230, dtype=np.uint8)

    x_center = 80
    vessel[:, x_center - 3 : x_center + 4] = 1
    av[:, x_center - 3 : x_center + 4] = 2
    rgb[:, x_center - 3 : x_center + 4, 1] = 40

    _write_mask(vessels_dir / "sample.png", vessel)
    _write_mask(av_dir / "sample.png", av)
    if needs_rgb:
        _write_rgb(rgb_dir / "sample.png", rgb)

    geometry_path = tmp_path / "disc_geometry.csv"
    pd.DataFrame(
        {
            "x_disc_center": [80.0],
            "y_disc_center": [80.0],
            "disc_radius_px": [20.0],
        },
        index=["sample"],
    ).to_csv(geometry_path)

    df_widths = measure_vessel_widths_between_disc_circle_pair(
        vessels_dir=vessels_dir,
        av_dir=av_dir,
        disc_geometry_path=geometry_path,
        inner_circle=OverlayCircle(name="inner", diameter=2.0),
        outer_circle=OverlayCircle(name="outer", diameter=3.0),
        samples_per_connection=5,
        width_config=width_config,
        rgb_dir=rgb_dir if needs_rgb else None,
    )
    _, df_equivalents = compute_revised_crx_from_widths(df_widths)

    assert not df_widths.empty
    assert sorted(df_widths["width_method"].unique().tolist()) == [expected_method]
    assert df_widths["measurement_valid"].tolist() == [True] * len(df_widths)
    assert not df_equivalents.empty
    assert df_equivalents.iloc[0]["metric"] == "CRVE"
    assert np.isfinite(df_equivalents.iloc[0]["equivalent_px"])
