from pathlib import Path

import logging
import pandas as pd
from click.testing import CliRunner
import torch
from PIL import Image

from vascx_models.cli import cli


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

    monkeypatch.setattr("vascx_models.cli.available_device_types", fake_available_device_types)
    monkeypatch.setattr("vascx_models.cli.resolve_device", fake_resolve_device)

    def fake_run_segmentation_vessels_and_av(**kwargs):
        calls["run_segmentation_vessels_and_av"] = kwargs

    def fake_run_segmentation_disc(**kwargs):
        calls["run_segmentation_disc"] = kwargs

    monkeypatch.setattr(
        "vascx_models.cli.run_segmentation_vessels_and_av",
        fake_run_segmentation_vessels_and_av,
    )
    monkeypatch.setattr("vascx_models.cli.run_segmentation_disc", fake_run_segmentation_disc)
    monkeypatch.setattr("vascx_models.cli.run_quality_estimation", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr("vascx_models.cli.run_fovea_detection", lambda **kwargs: pd.DataFrame())

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

    def fake_batch_create_overlays(**kwargs):
        calls.setdefault("batch_create_overlays", []).append(kwargs)

    monkeypatch.setattr("vascx_models.cli.generate_disc_circles", fake_generate_disc_circles)
    monkeypatch.setattr(
        "vascx_models.cli.measure_vessel_widths_between_disc_circle_pair",
        fake_measure_vessel_widths_between_disc_circle_pair,
    )
    monkeypatch.setattr("vascx_models.cli.batch_create_overlays", fake_batch_create_overlays)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "overlay:",
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
                "vessel_widths:",
                "  inner_circle: 2r",
                "  outer_circle: 3r",
                "  samples_per_connection: 4",
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
    assert calls["measure_vessel_widths"]["samples_per_connection"] == 4
    assert calls["generate_disc_circles"]["circles"][0].name == "2r"
    assert calls["generate_disc_circles"]["circles"][0].color == (0, 255, 0)
    overlay_calls = calls["batch_create_overlays"]
    assert len(overlay_calls) == 2
    assert overlay_calls[0]["output_dir"] == output_dir / "overlays"
    assert overlay_calls[0]["vessels_dir"] == output_dir / "vessels"
    measurement_data = overlay_calls[0]["vessel_width_data"]
    assert isinstance(measurement_data, pd.DataFrame)
    assert measurement_data.iloc[0]["width_px"] == 7.0
    assert overlay_calls[1]["output_dir"] == output_dir / "vessel_equivalent_overlays"
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
    monkeypatch.setattr("vascx_models.cli.run_quality_estimation", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr("vascx_models.cli.run_fovea_detection", lambda **kwargs: pd.DataFrame())

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
