import logging
from pathlib import Path

import pandas as pd
import torch
from click.testing import CliRunner
from PIL import Image

from vascx_models.cli import cli


def _write_minimal_vessel_metric_intermediates(source_dir: Path) -> None:
    vessels_dir = source_dir / "vessels"
    av_dir = source_dir / "artery_vein"
    preprocessed_dir = source_dir / "preprocessed_rgb"
    vessels_dir.mkdir(parents=True)
    av_dir.mkdir(parents=True)
    preprocessed_dir.mkdir()
    (vessels_dir / "sample.png").write_bytes(b"vessel")
    (av_dir / "sample.png").write_bytes(b"av")
    (preprocessed_dir / "sample.png").write_bytes(b"rgb")
    (source_dir / "quality.csv").write_text("image_id,q1,q2,q3\n", encoding="utf-8")
    (source_dir / "vessel_widths.csv").write_text("stale\n", encoding="utf-8")
    pd.DataFrame(
        {
            "x_disc_center": [16.0],
            "y_disc_center": [16.0],
            "disc_radius_px": [5.0],
        },
        index=["sample"],
    ).to_csv(source_dir / "disc_geometry.csv")


def test_cli_run_passes_measurement_config_and_data_to_overlays(
    tmp_path: Path, monkeypatch
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

    monkeypatch.setattr(
        "vascx_models.cli.available_device_types", fake_available_device_types
    )
    monkeypatch.setattr("vascx_models.cli.resolve_device", fake_resolve_device)

    def fake_run_segmentation_vessels_and_av(**kwargs):
        calls["run_segmentation_vessels_and_av"] = kwargs

    def fake_run_segmentation_disc(**kwargs):
        calls["run_segmentation_disc"] = kwargs

    monkeypatch.setattr(
        "vascx_models.cli.run_segmentation_vessels_and_av",
        fake_run_segmentation_vessels_and_av,
    )
    monkeypatch.setattr(
        "vascx_models.cli.run_segmentation_disc", fake_run_segmentation_disc
    )
    monkeypatch.setattr(
        "vascx_models.cli.run_quality_estimation", lambda **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        "vascx_models.cli.run_fovea_detection", lambda **kwargs: pd.DataFrame()
    )

    def fake_generate_disc_circles(**kwargs):
        calls["generate_disc_circles"] = kwargs
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
                }
            ]
        )
        df_tortuosities.to_csv(kwargs["output_path"], index=False)
        return df_tortuosities

    def fake_batch_create_overlays(**kwargs):
        calls.setdefault("batch_create_overlays", []).append(kwargs)

    monkeypatch.setattr(
        "vascx_models.cli.generate_disc_circles", fake_generate_disc_circles
    )
    monkeypatch.setattr(
        "vascx_models.cli.measure_vessel_widths_between_disc_circle_pair",
        fake_measure_vessel_widths_between_disc_circle_pair,
    )
    monkeypatch.setattr(
        "vascx_models.cli.measure_vessel_tortuosities_between_disc_circle_pair",
        fake_measure_vessel_tortuosities_between_disc_circle_pair,
    )
    monkeypatch.setattr(
        "vascx_models.cli.batch_create_overlays", fake_batch_create_overlays
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "overlay:",
                "  layers:",
                "    vessel_widths: false",
                "  colours:",
                "    artery: '#AA0000'",
                "    vein: '#0000BB'",
                "    vessel: '#00CC00'",
                "    disc: '#DDDDDD'",
                "  circles:",
                "    - name: 2r",
                "      diameter: 2.0",
                "      color: '#00FF00'",
                "    - name: 3r",
                "      diameter: 3.0",
                "      color: '#00FF00'",
                "    - name: 5r",
                "      diameter: 5.0",
                "      color: '#00FF00'",
                "vessel_widths:",
                "  inner_circle: 2r",
                "  outer_circle: 3r",
                "  samples_per_connection: 4",
                "  method: profile",
                "  profile:",
                "    image_source: preprocessed_rgb",
                "    fallback_to_mask: true",
                "vessel_tortuosities:",
                "  inner_circle: 2r",
                "  outer_circle: 5r",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(input_dir),
            str(output_dir),
            "--config",
            str(config_path),
            "--no-preprocess",
            "--overlay",
            "--vessels",
            "--disc",
            "--no-quality",
            "--no-fovea",
        ],
    )

    assert result.exit_code == 0, result.output
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
    assert calls["measure_vessel_widths"]["width_config"].method == "profile"
    assert calls["measure_vessel_widths"]["rgb_dir"] == output_dir / "preprocessed_rgb"
    assert calls["generate_disc_circles"]["circles"][0].name == "2r"
    assert calls["generate_disc_circles"]["circles"][0].color == (0, 255, 0)
    overlay_calls = calls["batch_create_overlays"]
    assert len(overlay_calls) == 2
    assert overlay_calls[0]["output_dir"] == output_dir / "overlays"
    assert overlay_calls[0]["vessels_dir"] == output_dir / "vessels"
    assert overlay_calls[0]["overlay_config"].layers.vessel_widths is False
    measurement_data = overlay_calls[0]["vessel_width_data"]
    assert isinstance(measurement_data, pd.DataFrame)
    assert measurement_data.iloc[0]["width_px"] == 7.0
    assert overlay_calls[1]["output_dir"] == output_dir / "vessel_equivalent_overlays"
    assert overlay_calls[1]["overlay_config"].layers.vessel_widths is True
    selected_measurement_data = overlay_calls[1]["vessel_width_data"]
    assert isinstance(selected_measurement_data, pd.DataFrame)
    assert selected_measurement_data.iloc[0]["width_px"] == 7.0

    vessel_equivalents = pd.read_csv(output_dir / "vessel_equivalents.csv")
    assert vessel_equivalents.columns.tolist() == [
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
    assert vessel_equivalents.iloc[0]["metric"] == "CRAE"
    assert vessel_equivalents.iloc[0]["vessel_ids_used"] == "artery_1"
    assert vessel_equivalents.iloc[0]["n_vessels_used"] == 1


def test_cli_vessel_metrics_copies_source_output_and_writes_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "metrics"
    _write_minimal_vessel_metric_intermediates(source_dir)
    (source_dir / "disc").mkdir()
    (source_dir / "disc" / "sample.png").write_bytes(b"disc")
    (source_dir / "disc_circles").mkdir()
    (source_dir / "disc_circles" / "stale.txt").write_text("stale", encoding="utf-8")
    (source_dir / "overlays").mkdir()
    (source_dir / "overlays" / "stale.png").write_text("stale", encoding="utf-8")
    (source_dir / "vessel_equivalent_overlays").mkdir()
    (source_dir / "vessel_equivalent_overlays" / "stale.png").write_text(
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

    monkeypatch.setattr(
        "vascx_models.cli.measure_vessel_widths_between_disc_circle_pair",
        fake_measure_vessel_widths_between_disc_circle_pair,
    )
    monkeypatch.setattr(
        "vascx_models.cli.measure_vessel_tortuosities_between_disc_circle_pair",
        fake_measure_vessel_tortuosities_between_disc_circle_pair,
    )

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

    monkeypatch.setattr(
        "vascx_models.cli.generate_disc_circles", fake_generate_disc_circles
    )

    def fake_batch_create_overlays(**kwargs):
        calls.setdefault("batch_create_overlays", []).append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["output_dir"] / "fresh.txt").write_text("fresh", encoding="utf-8")

    monkeypatch.setattr(
        "vascx_models.cli.batch_create_overlays", fake_batch_create_overlays
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "overlay:",
                "  enabled: true",
                "  circles:",
                "    - name: 2r",
                "      diameter: 2.0",
                "    - name: 3r",
                "      diameter: 3.0",
                "vessel_widths:",
                "  inner_circle: 2r",
                "  outer_circle: 3r",
                "  samples_per_connection: 4",
                "  method: profile",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "vessel-metrics",
            str(source_dir),
            str(output_dir),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "vessels" / "sample.png").read_bytes() == b"vessel"
    assert (output_dir / "artery_vein" / "sample.png").read_bytes() == b"av"
    assert (output_dir / "preprocessed_rgb" / "sample.png").read_bytes() == b"rgb"
    assert (output_dir / "quality.csv").read_text(encoding="utf-8") == (
        "image_id,q1,q2,q3\n"
    )
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
    assert calls["measure_vessel_widths"]["disc_geometry_path"] == (
        output_dir / "disc_geometry.csv"
    )
    assert calls["measure_vessel_widths"]["samples_per_connection"] == 4
    assert calls["measure_vessel_widths"]["width_config"].method == "profile"
    assert calls["measure_vessel_widths"]["rgb_dir"] == output_dir / "preprocessed_rgb"
    assert calls["measure_vessel_tortuosities"]["vessels_dir"] == (
        output_dir / "vessels"
    )
    assert calls["measure_vessel_tortuosities"]["av_dir"] == (
        output_dir / "artery_vein"
    )
    assert len(calls["batch_create_overlays"]) == 2
    assert calls["batch_create_overlays"][0]["output_dir"] == output_dir / "overlays"
    assert calls["batch_create_overlays"][1]["output_dir"] == (
        output_dir / "vessel_equivalent_overlays"
    )
    assert not (output_dir / "overlays" / "stale.png").exists()
    assert not (output_dir / "vessel_equivalent_overlays" / "stale.png").exists()
    assert (output_dir / "overlays" / "fresh.txt").exists()
    assert (output_dir / "vessel_equivalent_overlays" / "fresh.txt").exists()
    assert (output_dir / "vessel_widths.csv").exists()
    assert pd.read_csv(output_dir / "vessel_widths.csv").iloc[0]["width_px"] == 7.0
    assert (output_dir / "vessel_tortuosities.csv").exists()
    equivalents = pd.read_csv(output_dir / "vessel_equivalents.csv")
    assert equivalents.iloc[0]["metric"] == "CRAE"
    assert "mean_tortuosity_used" not in equivalents.columns


def test_cli_vessel_metrics_uses_timestamped_output_when_omitted(
    tmp_path: Path, monkeypatch
) -> None:
    source_dir = tmp_path / "source"
    _write_minimal_vessel_metric_intermediates(source_dir)
    created_output_dir = tmp_path / "output_20260427_123456"
    calls: dict[str, object] = {}

    class FixedDatetime:
        @classmethod
        def now(cls):
            return cls()

        def strftime(self, fmt):
            assert fmt == "%Y%m%d_%H%M%S"
            return "20260427_123456"

    def fake_compute_and_save_vessel_metrics(**kwargs):
        calls["compute_vessel_metrics"] = kwargs
        (kwargs["output_path"] / "vessel_widths.csv").write_text(
            "image_id\n", encoding="utf-8"
        )
        return tuple(pd.DataFrame() for _ in range(4))

    monkeypatch.setattr("vascx_models.cli.datetime", FixedDatetime)
    monkeypatch.setattr(
        "vascx_models.cli._compute_and_save_vessel_metrics",
        fake_compute_and_save_vessel_metrics,
    )
    monkeypatch.setattr(
        "vascx_models.cli._refresh_vessel_metric_overlays",
        lambda **kwargs: calls.setdefault("refresh_overlays", kwargs),
    )

    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        result = CliRunner().invoke(cli, ["vessel-metrics", str(source_dir)])

    assert result.exit_code == 0, result.output
    assert created_output_dir.exists()
    assert (created_output_dir / "vessels" / "sample.png").read_bytes() == b"vessel"
    assert calls["refresh_overlays"]["output_path"] == created_output_dir
    assert (created_output_dir / "quality.csv").exists()
    assert calls["compute_vessel_metrics"]["output_path"] == created_output_dir


def test_cli_vessel_metrics_rejects_nonempty_output(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    (source_dir / "vessels").mkdir(parents=True)
    (source_dir / "artery_vein").mkdir()
    (source_dir / "disc_geometry.csv").write_text("image_id\n", encoding="utf-8")
    output_dir = tmp_path / "metrics"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    result = CliRunner().invoke(
        cli, ["vessel-metrics", str(source_dir), str(output_dir)]
    )

    assert result.exit_code != 0
    assert "not empty" in result.output


def test_cli_vessel_metrics_rejects_output_inside_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_minimal_vessel_metric_intermediates(source_dir)
    output_dir = source_dir / "nested_output"

    result = CliRunner().invoke(
        cli, ["vessel-metrics", str(source_dir), str(output_dir)]
    )

    assert result.exit_code != 0
    assert "inside SOURCE_OUTPUT_PATH" in result.output
    assert not output_dir.exists()


def test_cli_run_reports_missing_path_column_in_csv(tmp_path: Path, caplog) -> None:
    csv_path = tmp_path / "images.csv"
    output_dir = tmp_path / "output"
    csv_path.write_text("id\nsample\n", encoding="utf-8")

    runner = CliRunner()
    with caplog.at_level(logging.ERROR):
        result = runner.invoke(
            cli,
            [
                "run",
                str(csv_path),
                str(output_dir),
                "--no-preprocess",
                "--no-vessels",
                "--no-disc",
                "--no-quality",
                "--no-fovea",
                "--no-overlay",
            ],
        )

    assert result.exit_code == 0
    assert "CSV must contain a 'path' column" in caplog.text


def test_cli_run_accepts_explicit_device_and_logs_selection(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(input_dir / "sample.png")

    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "vascx_models.cli.available_device_types",
        lambda: {"cuda": False, "mps": False, "cpu": True},
    )

    def fake_resolve_device(device_name):
        calls["device_name"] = device_name
        return torch.device("cpu")

    monkeypatch.setattr("vascx_models.cli.resolve_device", fake_resolve_device)
    monkeypatch.setattr(
        "vascx_models.cli.run_quality_estimation", lambda **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        "vascx_models.cli.run_fovea_detection", lambda **kwargs: pd.DataFrame()
    )

    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            cli,
            [
                "run",
                str(input_dir),
                str(output_dir),
                "--no-preprocess",
                "--no-vessels",
                "--no-disc",
                "--no-quality",
                "--no-fovea",
                "--no-overlay",
                "--device",
                "cpu",
            ],
        )

    assert result.exit_code == 0, result.output
    assert calls["device_name"] == "cpu"
    assert "Device availability: cuda=False, mps=False, cpu=True" in caplog.text
    assert "Using requested device 'cpu': cpu" in caplog.text
