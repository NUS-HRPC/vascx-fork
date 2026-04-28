from pathlib import Path

import numpy as np
from PIL import Image

from vascx_models.config import OverlayColors, OverlayConfig, OverlayLayers
from vascx_models.utils import create_fundus_overlay


def test_create_fundus_overlay_draws_kept_measurement_segments(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    vessel_path = tmp_path / "vessel.png"
    rgb = np.full((40, 40, 3), 255, dtype=np.uint8)
    Image.fromarray(rgb).save(rgb_path)
    vessel = np.zeros((40, 40), dtype=np.uint8)
    vessel[20, 14:27] = 1
    Image.fromarray(vessel).save(vessel_path)

    output = create_fundus_overlay(
        rgb_path=str(rgb_path),
        vessel_path=str(vessel_path),
        vessel_width_measurements=[
            {
                "x": 20.0,
                "y": 20.0,
                "x_start": 10.0,
                "y_start": 20.0,
                "x_end": 30.0,
                "y_end": 20.0,
            }
        ],
    )

    assert tuple(output[20, 20]) == (0, 0, 0)
    assert tuple(output[20, 14]) == (0, 0, 0)
    assert tuple(output[20, 26]) == (0, 0, 0)
    assert tuple(output[20, 13]) == (255, 255, 255)
    assert tuple(output[20, 27]) == (255, 255, 255)
    assert tuple(output[19, 20]) == (255, 255, 255)
    assert tuple(output[21, 20]) == (255, 255, 255)


def test_create_fundus_overlay_respects_vessel_width_layer_and_color(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "rgb.png"
    vessel_path = tmp_path / "vessel.png"
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(rgb_path)
    vessel = np.zeros((40, 40), dtype=np.uint8)
    vessel[20, 14:27] = 1
    Image.fromarray(vessel).save(vessel_path)

    hidden_output = create_fundus_overlay(
        rgb_path=str(rgb_path),
        vessel_path=str(vessel_path),
        vessel_width_measurements=[
            {
                "x": 20.0,
                "y": 20.0,
                "x_start": 10.0,
                "y_start": 20.0,
                "x_end": 30.0,
                "y_end": 20.0,
            }
        ],
        overlay_config=OverlayConfig(layers=OverlayLayers(vessel_widths=False)),
    )
    assert tuple(hidden_output[20, 20]) == (0, 0, 0)

    colored_output = create_fundus_overlay(
        rgb_path=str(rgb_path),
        vessel_path=str(vessel_path),
        vessel_width_measurements=[
            {
                "x": 20.0,
                "y": 20.0,
                "x_start": 10.0,
                "y_start": 20.0,
                "x_end": 30.0,
                "y_end": 20.0,
            }
        ],
        overlay_config=OverlayConfig(
            colors=OverlayColors(vessel_width=(1, 2, 3)),
        ),
    )
    assert tuple(colored_output[20, 20]) == (1, 2, 3)
    assert tuple(colored_output[20, 14]) == (1, 2, 3)
    assert tuple(colored_output[20, 26]) == (1, 2, 3)
    assert tuple(colored_output[20, 13]) == (0, 0, 0)
    assert tuple(colored_output[20, 27]) == (0, 0, 0)
    assert tuple(colored_output[19, 20]) == (0, 0, 0)


def test_create_fundus_overlay_draws_tortuosity_skeleton_and_chord(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "rgb.png"
    vessel_path = tmp_path / "vessel.png"
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(rgb_path)
    vessel = np.zeros((40, 40), dtype=np.uint8)
    vessel[18:23, 20] = 1
    Image.fromarray(vessel).save(vessel_path)

    output = create_fundus_overlay(
        rgb_path=str(rgb_path),
        vessel_path=str(vessel_path),
        tortuosity_measurements=[
            {
                "x_start": 20.0,
                "y_start": 18.0,
                "x_end": 20.0,
                "y_end": 22.0,
            }
        ],
        overlay_config=OverlayConfig(colors=OverlayColors(vessel=(0, 255, 0))),
    )

    assert tuple(output[18, 20]) == (0, 255, 0)
    assert tuple(output[20, 20]) == (0, 255, 0)
    assert tuple(output[22, 20]) == (0, 255, 0)
    assert tuple(output[20, 19]) == (0, 0, 0)
    assert tuple(output[20, 21]) == (0, 0, 0)
