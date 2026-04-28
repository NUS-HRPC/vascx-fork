from pathlib import Path

import numpy as np
from PIL import Image

from vascx_models.config import (
    OverlayCircle,
    OverlayColors,
    OverlayConfig,
    OverlayLayers,
)
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
    vessel[20, 20:25] = 1
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
    assert tuple(output[20, 24]) == (0, 0, 0)
    assert tuple(output[20, 19]) == (0, 0, 0)
    assert tuple(output[20, 21]) == (0, 0, 0)


def test_create_fundus_overlay_draws_tortuosity_on_top_of_other_layers(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "rgb.png"
    av_path = tmp_path / "av.png"
    disc_path = tmp_path / "disc.png"
    vessel_path = tmp_path / "vessel.png"
    circle_path = tmp_path / "circle.png"

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(rgb_path)

    av = np.zeros((100, 100), dtype=np.uint8)
    av[10, 10] = 1
    av[10, 11] = 2
    Image.fromarray(av).save(av_path)

    disc = np.zeros((100, 100), dtype=np.uint8)
    disc[45:56, 45:56] = 1
    Image.fromarray(disc).save(disc_path)

    vessel = np.zeros((100, 100), dtype=np.uint8)
    vessel[45:56, 50] = 1
    Image.fromarray(vessel).save(vessel_path)

    circle = np.zeros((100, 100), dtype=np.uint8)
    circle[30, 30] = 1
    Image.fromarray(circle).save(circle_path)

    output = create_fundus_overlay(
        rgb_path=str(rgb_path),
        av_path=str(av_path),
        disc_path=str(disc_path),
        vessel_path=str(vessel_path),
        circle_paths={"2r": str(circle_path)},
        tortuosity_measurements=[
            {
                "x_start": 50.0,
                "y_start": 45.0,
                "x_end": 50.0,
                "y_end": 55.0,
            }
        ],
        fovea_location=(20, 50),
        overlay_config=OverlayConfig(
            circles=(OverlayCircle(name="2r", diameter=2.0, color=(9, 8, 7)),),
            colors=OverlayColors(
                artery=(255, 0, 0),
                vein=(0, 0, 255),
                vessel=(0, 255, 0),
                disc=(255, 255, 255),
                fovea=(255, 255, 0),
                vessel_width=(0, 0, 0),
            ),
        ),
    )

    assert tuple(output[10, 10]) == (255, 0, 0)
    assert tuple(output[10, 11]) == (0, 0, 255)
    assert tuple(output[20, 20]) == (0, 0, 0)
    assert tuple(output[30, 30]) == (9, 8, 7)
    assert tuple(output[50, 20]) == (255, 255, 0)
    assert tuple(output[50, 50]) == (0, 255, 0)
    assert tuple(output[10, 11]) == (0, 0, 255)
    assert tuple(output[20, 20]) == (0, 0, 0)
    assert tuple(output[30, 30]) == (9, 8, 7)
    assert tuple(output[50, 20]) == (255, 255, 0)
    assert tuple(output[50, 50]) == (0, 255, 0)
