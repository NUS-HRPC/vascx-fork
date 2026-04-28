from pathlib import Path

import pytest

from vascx_models.config import load_app_config


def test_load_app_config_accepts_aliases_and_colours(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "overlay:",
                "  enabled: false",
                "  layers:",
                "    artery: false",
                "  colours:",
                "    veins: '#112233'",
            ]
        ),
        encoding="utf-8",
    )

    app_config = load_app_config(config_path)

    assert app_config.source_path == config_path
    assert app_config.overlay.enabled is False
    assert app_config.overlay.layers.arteries is False
    assert app_config.overlay.layers.vessel_widths is False
    assert app_config.overlay.colors.vein == (17, 34, 51)
    assert app_config.overlay.colors.vessel == (255, 255, 255)
    assert app_config.overlay.colors.vessel_width == (0, 0, 0)
    assert [circle.name for circle in app_config.overlay.circles] == ["2r", "3r", "5r"]
    assert [circle.diameter for circle in app_config.overlay.circles] == [2.0, 3.0, 5.0]
    assert app_config.overlay.circles[0].color == (0, 255, 0)
    assert app_config.vessel_widths.enabled is True
    assert app_config.vessel_tortuosities.enabled is True
    assert app_config.vessel_widths.samples_per_connection == 5


def test_load_app_config_rejects_unknown_layer(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "overlay:",
                "  layers:",
                "    unknown_layer: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported overlay layer"):
        load_app_config(config_path)


def test_load_app_config_rejects_overlay_circle_declarations(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "overlay:",
                "  circles:",
                "    - name: repeated",
                "      diameter: 2.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported keys in 'overlay': circles"):
        load_app_config(config_path)


def test_load_app_config_accepts_vessel_width_sampling_options(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "overlay:",
                "  colours:",
                "    vessel: '#0A0B0C'",
                "    vessel_widths: [1, 2, 3]",
                "  circle_colours:",
                "    3r: '#112233'",
                "    5r: [4, 5, 6]",
                "vessel_widths:",
                "  enabled: false",
                "  inner_circle: 2r",
                "  outer_circle: 3r",
                "  samples_per_connection: 4",
                "vessel_tortuosities:",
                "  enabled: true",
                "  inner_circle: 3r",
                "  outer_circle: 5r",
            ]
        ),
        encoding="utf-8",
    )

    app_config = load_app_config(config_path)

    assert app_config.overlay.layers.vessel_widths is False
    assert app_config.overlay.colors.vessel == (10, 11, 12)
    assert app_config.overlay.colors.vessel_width == (1, 2, 3)
    assert app_config.vessel_widths.enabled is False
    assert app_config.vessel_widths.inner_circle == "2r"
    assert app_config.vessel_widths.outer_circle == "3r"
    assert app_config.vessel_widths.samples_per_connection == 4
    assert app_config.vessel_widths.method == "mask"
    assert app_config.vessel_tortuosities.enabled is True
    assert app_config.vessel_tortuosities.inner_circle == "3r"
    assert app_config.vessel_tortuosities.outer_circle == "5r"
    assert [circle.name for circle in app_config.overlay.circles] == ["3r", "5r"]
    assert [circle.color for circle in app_config.overlay.circles] == [
        (17, 34, 51),
        (4, 5, 6),
    ]


def test_load_app_config_skips_all_circles_when_metrics_disabled(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vessel_widths:",
                "  enabled: false",
                "vessel_tortuosities:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    app_config = load_app_config(config_path)

    assert app_config.overlay.circles == ()


def test_load_app_config_rejects_unused_circle_colour(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "overlay:",
                "  circle_colours:",
                "    7r: '#123456'",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlay.circle_colors contains entries"):
        load_app_config(config_path)


def test_load_app_config_rejects_non_derived_circle_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vessel_widths:",
                "  inner_circle: inner",
                "  outer_circle: 3r",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must use the '<multiplier>r' format"):
        load_app_config(config_path)


def test_load_app_config_accepts_profile_width_options(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vessel_widths:",
                "  boundary_tolerance_px: 2.0",
                "  method: profile",
                "  mask:",
                "    tangent_window_px: 12.0",
                "    measurement_step_px: 0.5",
                "    boundary_refinement_steps: 8",
                "    trace_padding_px: 3.0",
                "  profile:",
                "    image_source: custom_rgb",
                "    channel: blue",
                "    half_length_px: 18.0",
                "    sample_step_px: 0.5",
                "    smoothing_sigma_px: 0.0",
                "    boundary_method: half_depth",
                "    threshold_alpha: 0.4",
                "    min_contrast: 0.03",
                "    min_width_px: 2.0",
                "    max_width_px: 50.0",
                "    use_mask_guardrail: false",
                "    mask_guardrail_min_ratio: 0.5",
                "    mask_guardrail_max_ratio: 2.0",
                "    fallback_to_mask: true",
                "  pvbm_mask:",
                "    direction_lag_px: 7.0",
                "    max_asymmetry_px: 1.5",
                "    trace_step_px: 0.75",
                "    boundary_adjust_px: 0.25",
                "    trace_padding_px: 4.0",
            ]
        ),
        encoding="utf-8",
    )

    app_config = load_app_config(config_path)

    assert app_config.vessel_widths.method == "profile"
    assert app_config.vessel_widths.boundary_tolerance_px == 2.0
    assert app_config.vessel_widths.mask.tangent_window_px == 12.0
    assert app_config.vessel_widths.mask.measurement_step_px == 0.5
    assert app_config.vessel_widths.mask.boundary_refinement_steps == 8
    assert app_config.vessel_widths.mask.trace_padding_px == 3.0
    assert app_config.vessel_widths.profile.image_source == "custom_rgb"
    assert app_config.vessel_widths.profile.channel == "blue"
    assert app_config.vessel_widths.profile.sample_step_px == 0.5
    assert app_config.vessel_widths.profile.use_mask_guardrail is False
    assert app_config.vessel_widths.profile.fallback_to_mask is True
    assert app_config.vessel_widths.pvbm_mask.direction_lag_px == 7.0
    assert app_config.vessel_widths.pvbm_mask.max_asymmetry_px == 1.5
    assert app_config.vessel_widths.pvbm_mask.trace_step_px == 0.75
    assert app_config.vessel_widths.pvbm_mask.boundary_adjust_px == 0.25
    assert app_config.vessel_widths.pvbm_mask.trace_padding_px == 4.0


def test_load_app_config_rejects_invalid_vessel_width_samples_per_connection(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vessel_widths:",
                "  samples_per_connection: 0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vessel_widths.samples_per_connection"):
        load_app_config(config_path)


def test_load_app_config_accepts_negative_vessel_width_samples_per_connection(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vessel_widths:",
                "  samples_per_connection: -1",
            ]
        ),
        encoding="utf-8",
    )

    app_config = load_app_config(config_path)

    assert app_config.vessel_widths.samples_per_connection == -1


def test_load_app_config_rejects_invalid_vessel_width_method(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "vessel_widths:",
                "  method: unsupported",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vessel_widths.method"):
        load_app_config(config_path)
