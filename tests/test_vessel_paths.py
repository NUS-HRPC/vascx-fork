import numpy as np
import pytest

from vascx_models.vessel_paths import (
    interpolate_path_point,
    path_cumulative_lengths,
    trace_vessel_paths_between_disc_circle_pair,
    trace_vessel_tortuosity_paths_between_disc_circle_pair,
)


def test_path_cumulative_lengths_and_interpolation() -> None:
    path_xy = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 4.0],
        ]
    )

    lengths = path_cumulative_lengths(path_xy)

    assert lengths.tolist() == pytest.approx([0.0, 5.0, 8.0])
    assert interpolate_path_point(path_xy, lengths, -1.0).tolist() == pytest.approx(
        [0.0, 0.0]
    )
    assert interpolate_path_point(path_xy, lengths, 2.5).tolist() == pytest.approx(
        [1.5, 2.0]
    )
    assert interpolate_path_point(path_xy, lengths, 6.5).tolist() == pytest.approx(
        [4.5, 4.0]
    )
    assert interpolate_path_point(path_xy, lengths, 99.0).tolist() == pytest.approx(
        [6.0, 4.0]
    )


def test_trace_vessel_paths_between_disc_circle_pair_orders_inner_to_outer() -> None:
    height = width = 160
    vessel = np.zeros((height, width), dtype=bool)
    vessel[:, 80] = True
    disc_center_xy = np.array([80.0, 80.0], dtype=float)

    paths = trace_vessel_paths_between_disc_circle_pair(
        vessel_mask=vessel,
        disc_center_xy=disc_center_xy,
        inner_radius_px=40.0,
        outer_radius_px=60.0,
        boundary_tolerance_px=1.5,
    )

    assert len(paths) == 2
    assert [path.connection_index for path in paths] == [1, 2]
    for path in paths:
        start_distance = float(np.linalg.norm(path.path_xy[0] - disc_center_xy))
        end_distance = float(np.linalg.norm(path.path_xy[-1] - disc_center_xy))
        assert start_distance == pytest.approx(40.0)
        assert end_distance == pytest.approx(60.0)
        assert path.skeleton.sum() == len(path.path_xy)
        assert np.column_stack(np.nonzero(path.skeleton))[:, ::-1].tolist() == sorted(
            path.path_xy.astype(int).tolist()
        )


def test_trace_vessel_paths_between_disc_circle_pair_prunes_dead_end_branch() -> None:
    height = width = 160
    vessel = np.zeros((height, width), dtype=bool)
    vessel[:, 80] = True
    vessel[125, 80:91] = True
    disc_center_xy = np.array([80.0, 80.0], dtype=float)

    paths = trace_vessel_paths_between_disc_circle_pair(
        vessel_mask=vessel,
        disc_center_xy=disc_center_xy,
        inner_radius_px=40.0,
        outer_radius_px=60.0,
        boundary_tolerance_px=1.5,
    )

    assert len(paths) == 2
    for path in paths:
        assert set(path.path_xy[:, 0].tolist()) == {80.0}


def test_trace_vessel_paths_between_disc_circle_pair_keeps_inner_trunk_for_fork() -> None:
    height = width = 180
    vessel = np.zeros((height, width), dtype=bool)
    vessel[130:136, 90] = True
    vessel[135, 40:141] = True
    disc_center_xy = np.array([90.0, 90.0], dtype=float)

    paths = trace_vessel_paths_between_disc_circle_pair(
        vessel_mask=vessel,
        disc_center_xy=disc_center_xy,
        inner_radius_px=40.0,
        outer_radius_px=60.0,
        boundary_tolerance_px=1.5,
    )

    assert len(paths) == 1
    assert paths[0].path_xy[:, 0].tolist() == [90.0] * len(paths[0].path_xy)
    assert paths[0].path_xy[:, 1].tolist() == pytest.approx(
        [130.0, 131.0, 132.0, 133.0, 134.0, 135.0]
    )


def test_trace_vessel_tortuosity_paths_between_disc_circle_pair_keeps_one_to_many_paths() -> (
    None
):
    height = width = 180
    vessel = np.zeros((height, width), dtype=bool)
    vessel[130:136, 90] = True
    vessel[135, 40:141] = True
    disc_center_xy = np.array([90.0, 90.0], dtype=float)

    paths = trace_vessel_tortuosity_paths_between_disc_circle_pair(
        vessel_mask=vessel,
        disc_center_xy=disc_center_xy,
        inner_radius_px=40.0,
        outer_radius_px=60.0,
        boundary_tolerance_px=1.5,
    )

    assert len(paths) == 2
    assert sorted(path.path_xy[-1, 0] for path in paths) == [51.0, 129.0]
    assert all(path.path_xy[0].tolist() == [90.0, 130.0] for path in paths)
