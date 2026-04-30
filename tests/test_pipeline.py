import logging
from dataclasses import replace
from pathlib import Path

import click
import pandas as pd
import pytest
import torch
from PIL import Image

from vascx_models import pipeline as pipeline_ops
from vascx_models.config import load_app_config


def _write_minimal_vessel_metric_intermediates(source_dir: Path) -> None:
    vessels_dir = source_dir / "vessels"
    av_dir = source_dir / "artery_vein"
    preprocessed_dir = source_dir / "preprocessed_rgb"
    vessels_dir.mkdir(parents=True)
    av_dir.mkdir()
    preprocessed_dir.mkdir()
    (vessels_dir / "sample.png").write_bytes(b"vessel")
    (av_dir / "sample.png").write_bytes(b"av")
    (preprocessed_dir / "sample.png").write_bytes(b"rgb")
    (source_dir / "quality.csv").write_text("image_id,q1,q2,q3\n", encoding="utf-8")
    (source_dir / "bounds.csv").write_text("image_id,x0,y0,x1,y1\n", encoding="utf-8")
    (source_dir / "vessel_widths.csv").write_text("stale\n", encoding="utf-8")
    pd.DataFrame(
        {
            "x_disc_center": [16.0],
            "y_disc_center": [16.0],
            "disc_radius_px": [5.0],
        },
        index=["sample"],
    ).to_csv(source_dir / "disc_geometry.csv")


def _load_app_config(tmp_path: Path, lines: list[str]):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return load_app_config(config_path)


def _build_dependencies(**overrides) -> pipeline_ops.PipelineDependencies:
    def resolve_circle_pair(circles, inner_circle_name=None, outer_circle_name=None):
        circles_by_name = {circle.name: circle for circle in circles}
        if inner_circle_name is not None and outer_circle_name is not None:
            return (
                circles_by_name[inner_circle_name],
                circles_by_name[outer_circle_name],
            )
        sorted_circles = sorted(circles, key=lambda circle: circle.diameter)
        return sorted_circles[0], sorted_circles[1]

    base = pipeline_ops.PipelineDependencies(
        available_device_types=lambda: {"cuda": False, "mps": False, "cpu": True},
        resolve_device=lambda _device_name: torch.device("cpu"),
        run_quality_estimation=lambda **kwargs: pd.DataFrame(),
        run_fovea_detection=lambda **kwargs: pd.DataFrame(),
        run_segmentation_vessels_and_av=lambda **kwargs: None,
        run_segmentation_disc=lambda **kwargs: None,
        generate_disc_circles=lambda **kwargs: pd.DataFrame(),
        resolve_vessel_width_circle_pair=resolve_circle_pair,
        measure_vessel_widths_between_disc_circle_pair=lambda **kwargs: pd.DataFrame(),
        compute_revised_crx_from_widths=lambda _df: (pd.DataFrame(), pd.DataFrame()),
        measure_vessel_tortuosities_between_disc_circle_pair=lambda **kwargs: pd.DataFrame(),
        summarize_vessel_tortuosities=lambda df, output_path=None: df,
        measure_vessel_branching_between_disc_circle_pair=lambda **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
        ),
        select_vessel_width_measurements_for_equivalents=lambda df_widths, _df_connections: df_widths,
        batch_create_overlays=lambda **kwargs: None,
        run_preprocessing=lambda **kwargs: None,
    )
    return replace(base, **overrides)


def test_run_pipeline_passes_measurement_config_and_data_to_overlays(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(input_dir / "sample.png")

    calls: dict[str, object] = {}

    def fake_available_device_types():
        return {"cuda": False, "mps": False, "cpu": True}

    def fake_resolve_device(device_name):
        calls["device_name"] = device_name
        return torch.device("cpu")

    def fake_run_segmentation_vessels_and_av(**kwargs):
        calls["run_segmentation_vessels_and_av"] = kwargs

    def fake_run_segmentation_disc(**kwargs):
        calls["run_segmentation_disc"] = kwargs

    def fake_generate_disc_circles(**kwargs):
        calls["generate_disc_circles"] = kwargs
        kwargs["circle_output_dir"].mkdir(parents=True, exist_ok=True)
        for circle in kwargs["circles"]:
            (kwargs["circle_output_dir"] / circle.name).mkdir(exist_ok=True)
        pd.DataFrame(
            {
                "x_disc_center": [16.0],
                "y_disc_center": [16.0],
                "disc_radius_px": [5.0],
            },
            index=["sample"],
        ).to_csv(kwargs["measurements_path"])

    def fake_measure_vessel_widths_between_disc_circle_pair(**kwargs):
        calls["measure_vessel_widths"] = kwargs
        df = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "inner_circle": "2r",
                    "outer_circle": "3r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 15.0,
                    "connection_index": 1,
                    "sample_index": 1,
                    "x": 16.0,
                    "y": 12.0,
                    "width_px": 7.0,
                    "x_start": 13.0,
                    "y_start": 12.0,
                    "x_end": 19.0,
                    "y_end": 12.0,
                    "vessel_type": "artery",
                }
            ]
        )
        df.to_csv(kwargs["output_path"], index=False)
        return df

    def fake_compute_revised_crx_from_widths(_df_widths):
        return (
            pd.DataFrame(
                [
                    {
                        "image_id": "sample",
                        "vessel_type": "artery",
                        "connection_index": 1,
                        "mean_width_px": 7.0,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "image_id": "sample",
                        "metric": "CRAE",
                        "vessel_type": "artery",
                        "requested_n_largest": 6,
                        "n_vessels_available": 1,
                        "n_vessels_used": 1,
                        "vessel_ids_used": "artery_1",
                        "mean_widths_used_px": "7.0",
                        "equivalent_px": 7.0,
                    }
                ]
            ),
        )

    def fake_measure_vessel_tortuosities_between_disc_circle_pair(**kwargs):
        calls["measure_vessel_tortuosities"] = kwargs
        df_tortuosities = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "inner_circle": "2r",
                    "outer_circle": "5r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 25.0,
                    "connection_index": 1,
                    "x_start": 16.0,
                    "y_start": 10.0,
                    "x_end": 16.0,
                    "y_end": 15.0,
                    "path_length_px": 5.0,
                    "chord_length_px": 5.0,
                    "tortuosity": 1.0,
                    "vessel_type": "artery",
                }
            ]
        )
        df_tortuosities.to_csv(kwargs["output_path"], index=False)
        return df_tortuosities

    def fake_summarize_vessel_tortuosities(df_tortuosities, output_path=None):
        summary = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "metric": "TORTA",
                    "vessel_type": "artery",
                    "inner_circle": "2r",
                    "outer_circle": "5r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 25.0,
                    "n_segments": 1,
                    "n_start_points": 1,
                    "total_length_px": 5.0,
                    "mean_tortuosity_weighted": 1.0,
                }
            ]
        )
        if output_path is not None:
            summary.to_csv(output_path, index=False)
        return summary

    def fake_measure_vessel_branching_between_disc_circle_pair(**kwargs):
        calls["measure_vessel_branching"] = kwargs
        df_branching = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "inner_circle": "2r",
                    "outer_circle": "5r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 25.0,
                    "connection_index": 1,
                    "x_junction": 16.0,
                    "y_junction": 14.0,
                    "parent_width_px": 7.0,
                    "daughter_1_width_px": 5.0,
                    "daughter_2_width_px": 5.0,
                    "branching_angle_deg": 90.0,
                    "branching_coefficient": 50.0 / 49.0,
                    "vessel_type": "artery",
                }
            ]
        )
        df_widths = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "connection_index": 1,
                    "branch_role": "parent",
                    "sample_index": 1,
                    "x_junction": 16.0,
                    "y_junction": 14.0,
                    "x": 16.0,
                    "y": 12.0,
                    "width_px": 7.0,
                    "x_start": 13.0,
                    "y_start": 12.0,
                    "x_end": 19.0,
                    "y_end": 12.0,
                    "vessel_type": "artery",
                }
            ]
        )
        df_branching.to_csv(kwargs["output_path"], index=False)
        df_widths.to_csv(kwargs["widths_output_path"], index=False)
        return df_branching, df_widths

    def fake_batch_create_overlays(**kwargs):
        calls.setdefault("batch_create_overlays", []).append(kwargs)

    deps = _build_dependencies(
        available_device_types=fake_available_device_types,
        resolve_device=fake_resolve_device,
        run_segmentation_vessels_and_av=fake_run_segmentation_vessels_and_av,
        run_segmentation_disc=fake_run_segmentation_disc,
        generate_disc_circles=fake_generate_disc_circles,
        measure_vessel_widths_between_disc_circle_pair=fake_measure_vessel_widths_between_disc_circle_pair,
        compute_revised_crx_from_widths=fake_compute_revised_crx_from_widths,
        measure_vessel_tortuosities_between_disc_circle_pair=fake_measure_vessel_tortuosities_between_disc_circle_pair,
        summarize_vessel_tortuosities=fake_summarize_vessel_tortuosities,
        measure_vessel_branching_between_disc_circle_pair=fake_measure_vessel_branching_between_disc_circle_pair,
        batch_create_overlays=fake_batch_create_overlays,
    )
    app_config = _load_app_config(
        tmp_path,
        [
            "overlay:",
            "  colours:",
            "    artery: '#AA0000'",
            "    vein: '#0000BB'",
            "    vessel: '#00CC00'",
            "    disc: '#DDDDDD'",
            "  circle_colours:",
            "    2r: '#123456'",
            "    3r: '#654321'",
            "    5r: '#ABCDEF'",
            "vessel_widths:",
            "  enabled: true",
            "  inner_circle: 2r",
            "  outer_circle: 3r",
            "  samples_per_connection: 4",
            "  boundary_tolerance_px: 2.5",
            "  method: profile",
            "  mask:",
            "    tangent_window_px: 11.0",
            "    measurement_step_px: 0.5",
            "    boundary_refinement_steps: 9",
            "    trace_padding_px: 3.0",
            "  profile:",
            "    image_source: preprocessed_rgb",
            "    fallback_to_mask: true",
            "vessel_tortuosities:",
            "  enabled: true",
            "  inner_circle: 2r",
            "  outer_circle: 5r",
            "vessel_branching:",
            "  enabled: true",
            "  inner_circle: 2r",
            "  outer_circle: 5r",
            "  width_samples_per_branch: 2",
        ],
    )

    pipeline_ops.run_pipeline(
        data_path=input_dir,
        output_path=output_dir,
        app_config=app_config,
        deps=deps,
        preprocess=False,
        vessels=True,
        disc=True,
        quality=False,
        fovea=False,
        overlay=True,
        device_name="auto",
        n_jobs=4,
    )

    assert calls["device_name"] == "auto"
    assert calls["run_segmentation_vessels_and_av"]["artery_color"] == (170, 0, 0)
    assert calls["run_segmentation_vessels_and_av"]["vein_color"] == (0, 0, 187)
    assert calls["run_segmentation_vessels_and_av"]["vessel_color"] == (0, 204, 0)
    assert calls["run_segmentation_vessels_and_av"]["device"] == torch.device("cpu")
    assert calls["run_segmentation_disc"]["disc_color"] == (221, 221, 221)
    assert calls["measure_vessel_widths"]["inner_circle"].name == "2r"
    assert calls["measure_vessel_widths"]["outer_circle"].name == "3r"
    assert calls["measure_vessel_tortuosities"]["inner_circle"].name == "2r"
    assert calls["measure_vessel_tortuosities"]["outer_circle"].name == "5r"
    assert calls["measure_vessel_tortuosities"]["output_path"] == (
        output_dir / "vessel_tortuosities.csv"
    )
    assert calls["measure_vessel_widths"]["samples_per_connection"] == 4
    assert calls["measure_vessel_widths"]["boundary_tolerance_px"] == 2.5
    assert calls["measure_vessel_widths"]["tangent_window_px"] == 11.0
    assert calls["measure_vessel_widths"]["measurement_step_px"] == 0.5
    assert calls["measure_vessel_widths"]["boundary_refinement_steps"] == 9
    assert calls["measure_vessel_widths"]["trace_padding_px"] == 3.0
    assert calls["measure_vessel_widths"]["width_config"].method == "profile"
    assert calls["measure_vessel_widths"]["width_config"].mask.tangent_window_px == 11.0
    assert calls["measure_vessel_widths"]["rgb_dir"] == output_dir / "preprocessed_rgb"
    assert calls["measure_vessel_branching"]["inner_circle"].name == "2r"
    assert calls["measure_vessel_branching"]["outer_circle"].name == "5r"
    assert calls["measure_vessel_branching"]["output_path"] == (
        output_dir / "vessel_branching.csv"
    )
    assert calls["measure_vessel_branching"]["widths_output_path"] == (
        output_dir / "vessel_branching_widths.csv"
    )
    assert (
        calls["measure_vessel_branching"][
            "branching_config"
        ].width_samples_per_branch
        == 2
    )
    assert calls["generate_disc_circles"]["circles"][0].name == "2r"
    assert calls["generate_disc_circles"]["circles"][0].color == (18, 52, 86)
    overlay_calls = calls["batch_create_overlays"]
    assert len(overlay_calls) == 4
    assert overlay_calls[0]["output_dir"] == output_dir / "overlays"
    assert overlay_calls[0]["vessels_dir"] == output_dir / "vessels"
    assert overlay_calls[0]["overlay_config"].layers.vessel_widths is False
    measurement_data = overlay_calls[0]["vessel_width_data"]
    assert isinstance(measurement_data, pd.DataFrame)
    assert measurement_data.iloc[0]["width_px"] == 7.0
    assert overlay_calls[1]["output_dir"] == output_dir / "vessel_tortuosity_overlays"
    assert overlay_calls[1]["av_dir"] == output_dir / "artery_vein"
    assert overlay_calls[1]["disc_dir"] == output_dir / "disc"
    assert overlay_calls[1]["vessels_dir"] == output_dir / "vessels"
    assert overlay_calls[1]["circle_dirs"]["2r"] == output_dir / "disc_circles" / "2r"
    assert overlay_calls[1]["circle_dirs"]["5r"] == output_dir / "disc_circles" / "5r"
    assert set(overlay_calls[1]["circle_dirs"]) == {"2r", "5r"}
    assert [circle.name for circle in overlay_calls[1]["overlay_config"].circles] == [
        "2r",
        "5r",
    ]
    assert overlay_calls[1]["tortuosity_data"].iloc[0]["tortuosity"] == 1.0
    assert overlay_calls[1]["fovea_data"] is None
    assert overlay_calls[1]["overlay_config"].colors.vessel == (0, 255, 0)
    assert overlay_calls[1]["overlay_config"].layers.arteries is True
    assert overlay_calls[1]["overlay_config"].layers.veins is True
    assert overlay_calls[1]["overlay_config"].layers.disc is True
    assert overlay_calls[1]["overlay_config"].layers.fovea is True
    assert overlay_calls[2]["output_dir"] == output_dir / "vessel_width_overlays"
    assert overlay_calls[2]["overlay_config"].layers.vessel_widths is True
    assert overlay_calls[2]["circle_dirs"]["2r"] == output_dir / "disc_circles" / "2r"
    assert overlay_calls[2]["circle_dirs"]["3r"] == output_dir / "disc_circles" / "3r"
    assert set(overlay_calls[2]["circle_dirs"]) == {"2r", "3r"}
    assert [circle.name for circle in overlay_calls[2]["overlay_config"].circles] == [
        "2r",
        "3r",
    ]
    selected_measurement_data = overlay_calls[2]["vessel_width_data"]
    assert isinstance(selected_measurement_data, pd.DataFrame)
    assert selected_measurement_data.iloc[0]["width_px"] == 7.0
    assert overlay_calls[3]["output_dir"] == output_dir / "vessel_branching_overlays"
    assert overlay_calls[3]["branching_data"].iloc[0]["branching_angle_deg"] == 90.0
    assert overlay_calls[3]["branching_width_data"].iloc[0]["width_px"] == 7.0
    assert overlay_calls[3]["overlay_config"].layers.vessel_branching is True

    vessel_widths_summary = pd.read_csv(output_dir / "vessel_widths_summary.csv")
    vessel_tortuosity_summary = pd.read_csv(
        output_dir / "vessel_tortuosity_summary.csv"
    )
    assert vessel_widths_summary.columns.tolist() == [
        "image_id",
        "metric",
        "vessel_type",
        "requested_n_largest",
        "n_vessels_available",
        "n_vessels_used",
        "vessel_ids_used",
        "mean_widths_used_px",
        "equivalent_px",
    ]
    assert vessel_widths_summary.iloc[0]["metric"] == "CRAE"
    assert vessel_widths_summary.iloc[0]["vessel_ids_used"] == "artery_1"
    assert vessel_widths_summary.iloc[0]["n_vessels_used"] == 1
    assert vessel_tortuosity_summary.columns.tolist() == [
        "image_id",
        "metric",
        "vessel_type",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
        "n_segments",
        "n_start_points",
        "total_length_px",
        "mean_tortuosity_weighted",
    ]
    assert vessel_tortuosity_summary.iloc[0]["metric"] == "TORTA"


def test_run_vessel_metrics_pipeline_copies_source_output_and_writes_outputs(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "metrics"
    _write_minimal_vessel_metric_intermediates(source_dir)
    (source_dir / "disc").mkdir()
    (source_dir / "disc" / "sample.png").write_bytes(b"disc")
    pd.DataFrame(
        {
            "x_fovea": [12],
            "y_fovea": [13],
        },
        index=["sample"],
    ).to_csv(source_dir / "fovea.csv")
    (source_dir / "disc_circles").mkdir()
    (source_dir / "disc_circles" / "stale.txt").write_text("stale", encoding="utf-8")
    (source_dir / "overlays").mkdir()
    (source_dir / "overlays" / "stale.png").write_text("stale", encoding="utf-8")
    (source_dir / "vessel_tortuosity_overlays").mkdir()
    (source_dir / "vessel_tortuosity_overlays" / "stale.png").write_text(
        "stale", encoding="utf-8"
    )
    (source_dir / "vessel_width_overlays").mkdir()
    (source_dir / "vessel_width_overlays" / "stale.png").write_text(
        "stale", encoding="utf-8"
    )
    (source_dir / "vessel_branching_overlays").mkdir()
    (source_dir / "vessel_branching_overlays" / "stale.png").write_text(
        "stale", encoding="utf-8"
    )

    calls: dict[str, object] = {}

    def fake_measure_vessel_widths_between_disc_circle_pair(**kwargs):
        calls["measure_vessel_widths"] = kwargs
        df_widths = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "inner_circle": "2r",
                    "outer_circle": "3r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 15.0,
                    "connection_index": 1,
                    "sample_index": 1,
                    "x": 16.0,
                    "y": 12.0,
                    "width_px": 7.0,
                    "x_start": 13.0,
                    "y_start": 12.0,
                    "x_end": 19.0,
                    "y_end": 12.0,
                    "vessel_type": "artery",
                },
                {
                    "image_id": "sample",
                    "inner_circle": "2r",
                    "outer_circle": "3r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 15.0,
                    "connection_index": 2,
                    "sample_index": 1,
                    "x": 20.0,
                    "y": 12.0,
                    "width_px": 8.0,
                    "x_start": 17.0,
                    "y_start": 12.0,
                    "x_end": 23.0,
                    "y_end": 12.0,
                    "vessel_type": "artery",
                },
            ]
        )
        df_widths.to_csv(kwargs["output_path"], index=False)
        return df_widths

    def fake_compute_revised_crx_from_widths(_df_widths):
        return (
            pd.DataFrame(
                [
                    {
                        "image_id": "sample",
                        "vessel_type": "artery",
                        "connection_index": 1,
                        "mean_width_px": 7.0,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "image_id": "sample",
                        "metric": "CRAE",
                        "vessel_type": "artery",
                        "requested_n_largest": 6,
                        "n_vessels_available": 2,
                        "n_vessels_used": 1,
                        "vessel_ids_used": "artery_1",
                        "mean_widths_used_px": "7.0",
                        "equivalent_px": 7.0,
                    }
                ]
            ),
        )

    def fake_measure_vessel_tortuosities_between_disc_circle_pair(**kwargs):
        calls["measure_vessel_tortuosities"] = kwargs
        df_tortuosities = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "inner_circle": "2r",
                    "outer_circle": "3r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 15.0,
                    "connection_index": 1,
                    "x_start": 16.0,
                    "y_start": 10.0,
                    "x_end": 16.0,
                    "y_end": 15.0,
                    "path_length_px": 5.0,
                    "chord_length_px": 5.0,
                    "tortuosity": 1.0,
                    "vessel_type": "artery",
                },
                {
                    "image_id": "sample",
                    "inner_circle": "2r",
                    "outer_circle": "3r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 15.0,
                    "connection_index": 2,
                    "x_start": 20.0,
                    "y_start": 10.0,
                    "x_end": 20.0,
                    "y_end": 16.0,
                    "path_length_px": 7.2,
                    "chord_length_px": 6.0,
                    "tortuosity": 1.2,
                    "vessel_type": "artery",
                },
            ]
        )
        df_tortuosities.to_csv(kwargs["output_path"], index=False)
        return df_tortuosities

    def fake_summarize_vessel_tortuosities(df_tortuosities, output_path=None):
        summary = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "metric": "TORTA",
                    "vessel_type": "artery",
                    "inner_circle": "2r",
                    "outer_circle": "3r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 15.0,
                    "n_segments": 2,
                    "n_start_points": 2,
                    "total_length_px": 12.2,
                    "mean_tortuosity_weighted": 1.118,
                }
            ]
        )
        if output_path is not None:
            summary.to_csv(output_path, index=False)
        return summary

    def fake_measure_vessel_branching_between_disc_circle_pair(**kwargs):
        calls["measure_vessel_branching"] = kwargs
        df_branching = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "inner_circle": "2r",
                    "outer_circle": "3r",
                    "inner_circle_radius_px": 10.0,
                    "outer_circle_radius_px": 15.0,
                    "connection_index": 1,
                    "x_junction": 16.0,
                    "y_junction": 14.0,
                    "parent_width_px": 7.0,
                    "daughter_1_width_px": 5.0,
                    "daughter_2_width_px": 5.0,
                    "branching_angle_deg": 90.0,
                    "branching_coefficient": 50.0 / 49.0,
                    "vessel_type": "artery",
                }
            ]
        )
        df_widths = pd.DataFrame(
            [
                {
                    "image_id": "sample",
                    "connection_index": 1,
                    "branch_role": "parent",
                    "sample_index": 1,
                    "x_junction": 16.0,
                    "y_junction": 14.0,
                    "x": 16.0,
                    "y": 12.0,
                    "width_px": 7.0,
                    "x_start": 13.0,
                    "y_start": 12.0,
                    "x_end": 19.0,
                    "y_end": 12.0,
                    "vessel_type": "artery",
                }
            ]
        )
        df_branching.to_csv(kwargs["output_path"], index=False)
        df_widths.to_csv(kwargs["widths_output_path"], index=False)
        return df_branching, df_widths

    def fake_generate_disc_circles(**kwargs):
        calls["generate_disc_circles"] = kwargs
        kwargs["circle_output_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["circle_output_dir"] / "fresh.txt").write_text(
            "fresh", encoding="utf-8"
        )
        pd.DataFrame(
            {
                "x_disc_center": [16.0],
                "y_disc_center": [16.0],
                "disc_radius_px": [5.0],
                "circle_2r_px": [10.0],
                "circle_3r_px": [15.0],
            },
            index=["sample"],
        ).to_csv(kwargs["measurements_path"])

    def fake_batch_create_overlays(**kwargs):
        calls.setdefault("batch_create_overlays", []).append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["output_dir"] / "fresh.txt").write_text("fresh", encoding="utf-8")

    deps = _build_dependencies(
        measure_vessel_widths_between_disc_circle_pair=fake_measure_vessel_widths_between_disc_circle_pair,
        compute_revised_crx_from_widths=fake_compute_revised_crx_from_widths,
        measure_vessel_tortuosities_between_disc_circle_pair=fake_measure_vessel_tortuosities_between_disc_circle_pair,
        summarize_vessel_tortuosities=fake_summarize_vessel_tortuosities,
        measure_vessel_branching_between_disc_circle_pair=fake_measure_vessel_branching_between_disc_circle_pair,
        generate_disc_circles=fake_generate_disc_circles,
        batch_create_overlays=fake_batch_create_overlays,
    )
    app_config = _load_app_config(
        tmp_path,
        [
            "overlay:",
            "  enabled: true",
            "vessel_widths:",
            "  enabled: true",
            "  inner_circle: 2r",
            "  outer_circle: 3r",
            "  samples_per_connection: 4",
            "  method: profile",
            "vessel_tortuosities:",
            "  enabled: true",
            "  inner_circle: 2r",
            "  outer_circle: 3r",
            "vessel_branching:",
            "  enabled: true",
            "  inner_circle: 2r",
            "  outer_circle: 3r",
        ],
    )

    pipeline_ops.run_vessel_metrics_pipeline(
        source_output_path=source_dir,
        output_path=output_dir,
        app_config=app_config,
        deps=deps,
        default_timestamped_output_path=lambda: tmp_path / "unused",
    )

    assert (output_dir / "vessels" / "sample.png").read_bytes() == b"vessel"
    assert (output_dir / "artery_vein" / "sample.png").read_bytes() == b"av"
    assert (output_dir / "preprocessed_rgb" / "sample.png").read_bytes() == b"rgb"
    assert (output_dir / "disc" / "sample.png").read_bytes() == b"disc"
    assert not (output_dir / "quality.csv").exists()
    assert not (output_dir / "bounds.csv").exists()
    assert (output_dir / "vessel_widths.csv").read_text(encoding="utf-8") != "stale\n"
    assert (output_dir / "fovea.csv").exists()
    assert calls["generate_disc_circles"]["disc_dir"] == output_dir / "disc"
    assert (
        calls["generate_disc_circles"]["circle_output_dir"]
        == output_dir / "disc_circles"
    )
    assert (output_dir / "disc_geometry.csv").exists()
    assert not (output_dir / "disc_circles" / "stale.txt").exists()
    assert (output_dir / "disc_circles" / "fresh.txt").exists()
    assert calls["measure_vessel_widths"]["vessels_dir"] == output_dir / "vessels"
    assert calls["measure_vessel_widths"]["av_dir"] == output_dir / "artery_vein"
    assert (
        calls["measure_vessel_widths"]["disc_geometry_path"]
        == output_dir / "disc_geometry.csv"
    )
    assert calls["measure_vessel_widths"]["samples_per_connection"] == 4
    assert calls["measure_vessel_widths"]["width_config"].method == "profile"
    assert calls["measure_vessel_widths"]["rgb_dir"] == output_dir / "preprocessed_rgb"
    assert calls["measure_vessel_tortuosities"]["vessels_dir"] == output_dir / "vessels"
    assert calls["measure_vessel_tortuosities"]["av_dir"] == output_dir / "artery_vein"
    assert calls["measure_vessel_branching"]["vessels_dir"] == output_dir / "vessels"
    assert calls["measure_vessel_branching"]["av_dir"] == output_dir / "artery_vein"
    assert len(calls["batch_create_overlays"]) == 4
    assert calls["batch_create_overlays"][0]["output_dir"] == output_dir / "overlays"
    assert calls["batch_create_overlays"][0]["fovea_data"] == {"sample": (12, 13)}
    assert (
        calls["batch_create_overlays"][1]["output_dir"]
        == output_dir / "vessel_tortuosity_overlays"
    )
    assert (
        calls["batch_create_overlays"][2]["output_dir"]
        == output_dir / "vessel_width_overlays"
    )
    assert (
        calls["batch_create_overlays"][3]["output_dir"]
        == output_dir / "vessel_branching_overlays"
    )
    assert not (output_dir / "overlays" / "stale.png").exists()
    assert not (output_dir / "vessel_width_overlays" / "stale.png").exists()
    assert not (output_dir / "vessel_tortuosity_overlays" / "stale.png").exists()
    assert not (output_dir / "vessel_branching_overlays" / "stale.png").exists()
    assert (output_dir / "overlays" / "fresh.txt").exists()
    assert (output_dir / "vessel_width_overlays" / "fresh.txt").exists()
    assert (output_dir / "vessel_tortuosity_overlays" / "fresh.txt").exists()
    assert (output_dir / "vessel_branching_overlays" / "fresh.txt").exists()
    assert (output_dir / "vessel_widths.csv").exists()
    assert pd.read_csv(output_dir / "vessel_widths.csv").iloc[0]["width_px"] == 7.0
    assert (output_dir / "vessel_tortuosities.csv").exists()
    assert (output_dir / "vessel_tortuosity_summary.csv").exists()
    assert (output_dir / "vessel_branching.csv").exists()
    assert (output_dir / "vessel_branching_widths.csv").exists()
    assert (output_dir / "vessel_tortuosity_overlays").exists()
    assert (output_dir / "vessel_branching_overlays").exists()
    widths_summary = pd.read_csv(output_dir / "vessel_widths_summary.csv")
    assert widths_summary.iloc[0]["metric"] == "CRAE"
    assert "mean_tortuosity_used" not in widths_summary.columns


def test_run_pipeline_skips_disabled_metric_sections(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(input_dir / "sample.png")

    calls: dict[str, object] = {}
    deps = _build_dependencies(
        run_segmentation_vessels_and_av=lambda **kwargs: calls.setdefault(
            "run_segmentation_vessels_and_av", kwargs
        ),
        run_segmentation_disc=lambda **kwargs: calls.setdefault(
            "run_segmentation_disc", kwargs
        ),
        generate_disc_circles=lambda **kwargs: calls.setdefault(
            "generate_disc_circles", kwargs
        ),
        measure_vessel_widths_between_disc_circle_pair=lambda **kwargs: calls.setdefault(
            "measure_vessel_widths", kwargs
        ),
        measure_vessel_tortuosities_between_disc_circle_pair=lambda **kwargs: calls.setdefault(
            "measure_vessel_tortuosities", kwargs
        ),
        measure_vessel_branching_between_disc_circle_pair=lambda **kwargs: calls.setdefault(
            "measure_vessel_branching", kwargs
        ),
        batch_create_overlays=lambda **kwargs: calls.setdefault(
            "batch_create_overlays", []
        ).append(kwargs),
    )
    app_config = _load_app_config(
        tmp_path,
        [
            "vessel_widths:",
            "  enabled: false",
            "vessel_tortuosities:",
            "  enabled: false",
            "vessel_branching:",
            "  enabled: false",
        ],
    )

    pipeline_ops.run_pipeline(
        data_path=input_dir,
        output_path=output_dir,
        app_config=app_config,
        deps=deps,
        preprocess=False,
        vessels=True,
        disc=True,
        quality=False,
        fovea=False,
        overlay=True,
        device_name="auto",
        n_jobs=4,
    )

    assert "generate_disc_circles" not in calls
    assert "measure_vessel_widths" not in calls
    assert "measure_vessel_tortuosities" not in calls
    assert "measure_vessel_branching" not in calls
    assert len(calls["batch_create_overlays"]) == 1
    assert calls["batch_create_overlays"][0]["output_dir"] == output_dir / "overlays"


def test_run_vessel_metrics_pipeline_uses_timestamped_output_when_omitted(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    _write_minimal_vessel_metric_intermediates(source_dir)
    created_output_dir = tmp_path / "output_20260427_123456"

    def fake_measure_vessel_widths_between_disc_circle_pair(**kwargs):
        (kwargs["output_path"] / "..").resolve()
        pd.DataFrame(columns=["image_id"]).to_csv(kwargs["output_path"], index=False)
        return pd.DataFrame(columns=["image_id"])

    def fake_compute_revised_crx_from_widths(_df_widths):
        return pd.DataFrame(), pd.DataFrame(columns=["image_id"])

    def fake_measure_vessel_tortuosities_between_disc_circle_pair(**kwargs):
        pd.DataFrame(columns=["image_id"]).to_csv(kwargs["output_path"], index=False)
        return pd.DataFrame(columns=["image_id"])

    def fake_summarize_vessel_tortuosities(_df_tortuosities, output_path=None):
        summary = pd.DataFrame(columns=["image_id"])
        if output_path is not None:
            summary.to_csv(output_path, index=False)
        return summary

    deps = _build_dependencies(
        measure_vessel_widths_between_disc_circle_pair=fake_measure_vessel_widths_between_disc_circle_pair,
        compute_revised_crx_from_widths=fake_compute_revised_crx_from_widths,
        measure_vessel_tortuosities_between_disc_circle_pair=fake_measure_vessel_tortuosities_between_disc_circle_pair,
        summarize_vessel_tortuosities=fake_summarize_vessel_tortuosities,
    )
    app_config = _load_app_config(tmp_path, ["overlay:", "  enabled: true"])

    pipeline_ops.run_vessel_metrics_pipeline(
        source_output_path=source_dir,
        output_path=None,
        app_config=app_config,
        deps=deps,
        default_timestamped_output_path=lambda: created_output_dir,
    )

    assert created_output_dir.exists()
    assert (created_output_dir / "vessels" / "sample.png").read_bytes() == b"vessel"
    assert not (created_output_dir / "quality.csv").exists()
    assert (created_output_dir / "preprocessed_rgb" / "sample.png").read_bytes() == b"rgb"


def test_run_vessel_metrics_pipeline_rejects_nonempty_output(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    (source_dir / "vessels").mkdir(parents=True)
    (source_dir / "artery_vein").mkdir()
    (source_dir / "disc_geometry.csv").write_text("image_id\n", encoding="utf-8")
    output_dir = tmp_path / "metrics"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(click.ClickException, match="not empty"):
        pipeline_ops.run_vessel_metrics_pipeline(
            source_output_path=source_dir,
            output_path=output_dir,
            app_config=object(),
            deps=_build_dependencies(),
            default_timestamped_output_path=lambda: tmp_path / "unused",
        )


def test_run_vessel_metrics_pipeline_rejects_output_inside_source(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    _write_minimal_vessel_metric_intermediates(source_dir)
    output_dir = source_dir / "nested_output"

    with pytest.raises(click.ClickException, match="inside SOURCE_OUTPUT_PATH"):
        pipeline_ops.run_vessel_metrics_pipeline(
            source_output_path=source_dir,
            output_path=output_dir,
            app_config=object(),
            deps=_build_dependencies(),
            default_timestamped_output_path=lambda: tmp_path / "unused",
        )
    assert not output_dir.exists()


def test_run_pipeline_reports_missing_path_column_in_csv(
    tmp_path: Path, caplog
) -> None:
    csv_path = tmp_path / "images.csv"
    output_dir = tmp_path / "output"
    csv_path.write_text("id\nsample\n", encoding="utf-8")
    app_config = _load_app_config(tmp_path, ["overlay:", "  enabled: false"])

    with caplog.at_level(logging.ERROR):
        pipeline_ops.run_pipeline(
            data_path=csv_path,
            output_path=output_dir,
            app_config=app_config,
            deps=_build_dependencies(),
            preprocess=False,
            vessels=False,
            disc=False,
            quality=False,
            fovea=False,
            overlay=False,
            device_name="auto",
            n_jobs=4,
        )

    assert "CSV must contain a 'path' column" in caplog.text


def test_run_pipeline_accepts_explicit_device_and_logs_selection(
    tmp_path: Path, caplog
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(input_dir / "sample.png")

    calls: dict[str, object] = {}

    def fake_resolve_device(device_name):
        calls["device_name"] = device_name
        return torch.device("cpu")

    deps = _build_dependencies(
        available_device_types=lambda: {"cuda": False, "mps": False, "cpu": True},
        resolve_device=fake_resolve_device,
    )
    app_config = _load_app_config(tmp_path, ["overlay:", "  enabled: false"])

    with caplog.at_level(logging.INFO):
        pipeline_ops.run_pipeline(
            data_path=input_dir,
            output_path=output_dir,
            app_config=app_config,
            deps=deps,
            preprocess=False,
            vessels=False,
            disc=False,
            quality=False,
            fovea=False,
            overlay=False,
            device_name="cpu",
            n_jobs=4,
        )

    assert calls["device_name"] == "cpu"
    assert "Device availability: cuda=False, mps=False, cpu=True" in caplog.text
    assert "Using requested device 'cpu': cpu" in caplog.text
