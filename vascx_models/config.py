from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import yaml

DEFAULT_CONFIG_NAME = "config.yaml"
DEFAULT_DERIVED_CIRCLE_COLOR = (0, 255, 0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class OverlayLayers:
    arteries: bool = True
    veins: bool = True
    disc: bool = True
    fovea: bool = True
    vessel_widths: bool = True


@dataclass(frozen=True)
class OverlayColors:
    artery: tuple[int, int, int] = (255, 0, 0)
    vein: tuple[int, int, int] = (0, 0, 255)
    vessel: tuple[int, int, int] = (255, 255, 255)
    disc: tuple[int, int, int] = (255, 255, 255)
    fovea: tuple[int, int, int] = (255, 255, 0)
    vessel_width: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class OverlayCircle:
    name: str
    diameter: float
    color: tuple[int, int, int] = (0, 0, 0)


def default_overlay_circles() -> tuple[OverlayCircle, ...]:
    return (
        OverlayCircle(name="2r", diameter=2.0, color=DEFAULT_DERIVED_CIRCLE_COLOR),
        OverlayCircle(name="3r", diameter=3.0, color=DEFAULT_DERIVED_CIRCLE_COLOR),
        OverlayCircle(name="5r", diameter=5.0, color=DEFAULT_DERIVED_CIRCLE_COLOR),
    )


@dataclass(frozen=True)
class OverlayConfig:
    enabled: bool = True
    layers: OverlayLayers = field(default_factory=OverlayLayers)
    colors: OverlayColors = field(default_factory=OverlayColors)
    circles: tuple[OverlayCircle, ...] = field(default_factory=default_overlay_circles)


@dataclass(frozen=True)
class VesselWidthConfig:
    enabled: bool = True
    inner_circle: str | None = "2r"
    outer_circle: str | None = "3r"
    samples_per_connection: int = 5
    boundary_tolerance_px: float = 1.5
    method: str = "mask"
    mask: "MaskWidthConfig" = field(default_factory=lambda: MaskWidthConfig())
    pvbm_mask: "PVBMMaskWidthConfig" = field(
        default_factory=lambda: PVBMMaskWidthConfig()
    )
    profile: "ProfileWidthConfig" = field(default_factory=lambda: ProfileWidthConfig())


@dataclass(frozen=True)
class VesselTortuosityConfig:
    enabled: bool = True
    inner_circle: str | None = "2r"
    outer_circle: str | None = "5r"


@dataclass(frozen=True)
class PVBMMaskWidthConfig:
    direction_lag_px: float = 6.0
    max_asymmetry_px: float = 1.0
    trace_step_px: float = 1.0
    boundary_adjust_px: float = 0.5
    trace_padding_px: float = 2.0


@dataclass(frozen=True)
class MaskWidthConfig:
    tangent_window_px: float = 10.0
    measurement_step_px: float = 0.25
    boundary_refinement_steps: int = 12
    trace_padding_px: float = 2.0


@dataclass(frozen=True)
class ProfileWidthConfig:
    image_source: str = "preprocessed_rgb"
    channel: str = "green"
    half_length_px: float = 20.0
    sample_step_px: float = 0.25
    smoothing_sigma_px: float = 1.0
    boundary_method: str = "half_depth"
    threshold_alpha: float = 0.5
    min_contrast: float = 0.05
    min_width_px: float = 1.0
    max_width_px: float = 80.0
    use_mask_guardrail: bool = True
    mask_guardrail_min_ratio: float = 0.4
    mask_guardrail_max_ratio: float = 2.5
    fallback_to_mask: bool = False
    tangent_window_px: float = 10.0


@dataclass(frozen=True)
class AppConfig:
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    vessel_widths: VesselWidthConfig = field(default_factory=VesselWidthConfig)
    vessel_tortuosities: VesselTortuosityConfig = field(
        default_factory=VesselTortuosityConfig
    )
    source_path: Path | None = None


def default_config_candidates() -> list[Path]:
    candidates = [Path.cwd() / DEFAULT_CONFIG_NAME, _repo_root() / DEFAULT_CONFIG_NAME]
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique_candidates.append(candidate)
            seen.add(resolved)
    return unique_candidates


def resolve_config_path(config_path: str | Path | None) -> Path | None:
    if config_path is not None:
        candidate = Path(config_path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Config file not found: {candidate}")
        return candidate

    for candidate in default_config_candidates():
        if candidate.exists():
            return candidate
    return None


def load_app_config(config_path: str | Path | None = None) -> AppConfig:
    resolved_path = resolve_config_path(config_path)
    if resolved_path is None:
        return AppConfig()

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("Config root must be a mapping")

    overlay_raw = raw_config.get("overlay", {})
    if overlay_raw is None:
        overlay_raw = {}
    if not isinstance(overlay_raw, dict):
        raise ValueError("'overlay' must be a mapping")

    layer_overrides = overlay_raw.get("layers", {})
    if layer_overrides is None:
        layer_overrides = {}
    if not isinstance(layer_overrides, dict):
        raise ValueError("'overlay.layers' must be a mapping")

    color_overrides = overlay_raw.get("colors", overlay_raw.get("colours", {}))
    if color_overrides is None:
        color_overrides = {}
    if not isinstance(color_overrides, dict):
        raise ValueError("'overlay.colors' must be a mapping")

    circle_color_overrides = overlay_raw.get(
        "circle_colors", overlay_raw.get("circle_colours", {})
    )
    if circle_color_overrides is None:
        circle_color_overrides = {}
    if not isinstance(circle_color_overrides, Mapping):
        raise ValueError("'overlay.circle_colors' must be a mapping")

    vessel_widths_raw = raw_config.get("vessel_widths", {})
    if vessel_widths_raw is None:
        vessel_widths_raw = {}
    if not isinstance(vessel_widths_raw, dict):
        raise ValueError("'vessel_widths' must be a mapping")

    vessel_tortuosities_raw = raw_config.get("vessel_tortuosities", {})
    if vessel_tortuosities_raw is None:
        vessel_tortuosities_raw = {}
    if not isinstance(vessel_tortuosities_raw, dict):
        raise ValueError("'vessel_tortuosities' must be a mapping")

    unsupported_overlay_keys = set(overlay_raw) - {
        "enabled",
        "layers",
        "colors",
        "colours",
        "circle_colors",
        "circle_colours",
    }
    if unsupported_overlay_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_overlay_keys))
        raise ValueError(f"Unsupported keys in 'overlay': {unsupported}")

    overlay_layers = _build_overlay_layers(layer_overrides)
    vessel_widths = _build_vessel_width_config(vessel_widths_raw)
    vessel_tortuosities = _build_vessel_tortuosity_config(vessel_tortuosities_raw)
    overlay_colors = _build_overlay_colors(color_overrides)
    circles = _derive_overlay_circles(
        vessel_widths,
        vessel_tortuosities,
        circle_color_overrides=_build_overlay_circle_colors(circle_color_overrides),
    )

    return AppConfig(
        overlay=OverlayConfig(
            enabled=_coerce_bool(overlay_raw.get("enabled", True), "overlay.enabled"),
            layers=overlay_layers,
            colors=overlay_colors,
            circles=circles,
        ),
        vessel_widths=vessel_widths,
        vessel_tortuosities=vessel_tortuosities,
        source_path=resolved_path,
    )


def _build_overlay_layers(raw_layers: Mapping[str, object]) -> OverlayLayers:
    defaults = OverlayLayers()
    alias_map = {
        "artery": "arteries",
        "arteries": "arteries",
        "vein": "veins",
        "veins": "veins",
        "disc": "disc",
        "fovea": "fovea",
    }
    values = defaults.__dict__.copy()
    values["vessel_widths"] = False
    for raw_key, raw_value in raw_layers.items():
        if raw_key not in alias_map:
            raise ValueError(f"Unsupported overlay layer '{raw_key}'")
        normalized_key = alias_map[raw_key]
        values[normalized_key] = _coerce_bool(raw_value, f"overlay.layers.{raw_key}")
    return OverlayLayers(**values)


def _build_overlay_colors(raw_colors: Mapping[str, object]) -> OverlayColors:
    defaults = OverlayColors()
    alias_map = {
        "artery": "artery",
        "arteries": "artery",
        "vein": "vein",
        "veins": "vein",
        "vessel": "vessel",
        "vessels": "vessel",
        "disc": "disc",
        "fovea": "fovea",
        "vessel_width": "vessel_width",
        "vessel_widths": "vessel_width",
    }
    values = defaults.__dict__.copy()
    for raw_key, raw_value in raw_colors.items():
        if raw_key not in alias_map:
            raise ValueError(f"Unsupported overlay color '{raw_key}'")
        normalized_key = alias_map[raw_key]
        values[normalized_key] = _parse_rgb(raw_value, f"overlay.colors.{raw_key}")
    return OverlayColors(**values)


def _build_overlay_circle_colors(
    raw_circle_colors: Mapping[str, object],
) -> dict[str, tuple[int, int, int]]:
    circle_colors: dict[str, tuple[int, int, int]] = {}
    for raw_key, raw_value in raw_circle_colors.items():
        circle_name = _coerce_circle_name(raw_key, "overlay.circle_colors")
        circle_colors[circle_name] = _parse_rgb(
            raw_value, f"overlay.circle_colors.{circle_name}"
        )
    return circle_colors


def _derive_overlay_circles(
    vessel_widths: VesselWidthConfig,
    vessel_tortuosities: VesselTortuosityConfig,
    circle_color_overrides: Mapping[str, tuple[int, int, int]] | None = None,
) -> tuple[OverlayCircle, ...]:
    circles_by_name: dict[str, OverlayCircle] = {}
    circle_color_overrides = circle_color_overrides or {}
    for section_enabled, field_name, circle_name in (
        (
            vessel_widths.enabled,
            "vessel_widths.inner_circle",
            vessel_widths.inner_circle,
        ),
        (
            vessel_widths.enabled,
            "vessel_widths.outer_circle",
            vessel_widths.outer_circle,
        ),
        (
            vessel_tortuosities.enabled,
            "vessel_tortuosities.inner_circle",
            vessel_tortuosities.inner_circle,
        ),
        (
            vessel_tortuosities.enabled,
            "vessel_tortuosities.outer_circle",
            vessel_tortuosities.outer_circle,
        ),
    ):
        if not section_enabled or circle_name is None:
            continue
        normalized_name = _coerce_circle_name(circle_name, field_name)
        circles_by_name.setdefault(
            normalized_name,
            OverlayCircle(
                name=normalized_name,
                diameter=_circle_name_to_diameter(normalized_name, field_name),
                color=circle_color_overrides.get(
                    normalized_name, DEFAULT_DERIVED_CIRCLE_COLOR
                ),
            ),
        )

    unknown_circle_colors = sorted(
        set(circle_color_overrides).difference(circles_by_name)
    )
    if unknown_circle_colors:
        unknown = ", ".join(unknown_circle_colors)
        raise ValueError(
            "overlay.circle_colors contains entries for disabled or undefined circles: "
            f"{unknown}"
        )

    circles = tuple(
        sorted(
            circles_by_name.values(), key=lambda circle: (circle.diameter, circle.name)
        )
    )
    return circles


def _build_vessel_width_config(
    raw_vessel_widths: Mapping[str, object],
) -> VesselWidthConfig:
    unsupported_keys = set(raw_vessel_widths) - {
        "enabled",
        "inner_circle",
        "outer_circle",
        "samples_per_connection",
        "boundary_tolerance_px",
        "method",
        "mask",
        "pvbm_mask",
        "profile",
    }
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise ValueError(f"Unsupported keys in 'vessel_widths': {unsupported}")

    pvbm_mask_raw = raw_vessel_widths.get("pvbm_mask", {})
    if pvbm_mask_raw is None:
        pvbm_mask_raw = {}
    if not isinstance(pvbm_mask_raw, Mapping):
        raise ValueError("'vessel_widths.pvbm_mask' must be a mapping")

    mask_raw = raw_vessel_widths.get("mask", {})
    if mask_raw is None:
        mask_raw = {}
    if not isinstance(mask_raw, Mapping):
        raise ValueError("'vessel_widths.mask' must be a mapping")

    profile_raw = raw_vessel_widths.get("profile", {})
    if profile_raw is None:
        profile_raw = {}
    if not isinstance(profile_raw, Mapping):
        raise ValueError("'vessel_widths.profile' must be a mapping")

    return VesselWidthConfig(
        enabled=_coerce_bool(
            raw_vessel_widths.get("enabled", True),
            "vessel_widths.enabled",
        ),
        inner_circle=_coerce_optional_string(
            raw_vessel_widths.get("inner_circle", "2r"),
            "vessel_widths.inner_circle",
        ),
        outer_circle=_coerce_optional_string(
            raw_vessel_widths.get("outer_circle", "3r"),
            "vessel_widths.outer_circle",
        ),
        samples_per_connection=_coerce_nonzero_int(
            raw_vessel_widths.get("samples_per_connection", 5),
            "vessel_widths.samples_per_connection",
        ),
        boundary_tolerance_px=_coerce_positive_float(
            raw_vessel_widths.get("boundary_tolerance_px", 1.5),
            "vessel_widths.boundary_tolerance_px",
        ),
        method=_coerce_choice(
            raw_vessel_widths.get("method", "mask"),
            "vessel_widths.method",
            {"mask", "pvbm_mask", "profile"},
        ),
        mask=_build_mask_width_config(mask_raw),
        pvbm_mask=_build_pvbm_mask_width_config(pvbm_mask_raw),
        profile=_build_profile_width_config(profile_raw),
    )


def _build_vessel_tortuosity_config(
    raw_vessel_tortuosities: Mapping[str, object],
) -> VesselTortuosityConfig:
    unsupported_keys = set(raw_vessel_tortuosities) - {
        "enabled",
        "inner_circle",
        "outer_circle",
    }
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise ValueError(f"Unsupported keys in 'vessel_tortuosities': {unsupported}")

    return VesselTortuosityConfig(
        enabled=_coerce_bool(
            raw_vessel_tortuosities.get("enabled", True),
            "vessel_tortuosities.enabled",
        ),
        inner_circle=_coerce_optional_string(
            raw_vessel_tortuosities.get("inner_circle", "2r"),
            "vessel_tortuosities.inner_circle",
        ),
        outer_circle=_coerce_optional_string(
            raw_vessel_tortuosities.get("outer_circle", "5r"),
            "vessel_tortuosities.outer_circle",
        ),
    )


def _circle_name_to_diameter(circle_name: str, field_name: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)r", circle_name)
    if match is None:
        raise ValueError(
            f"'{field_name}' must use the '<multiplier>r' format, for example '2r' or '3.5r'"
        )
    return _coerce_positive_float(float(match.group(1)), field_name)


def _build_pvbm_mask_width_config(
    raw_pvbm_mask: Mapping[str, object],
) -> PVBMMaskWidthConfig:
    unsupported_keys = set(raw_pvbm_mask) - {
        "direction_lag_px",
        "max_asymmetry_px",
        "trace_step_px",
        "boundary_adjust_px",
        "trace_padding_px",
    }
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise ValueError(
            f"Unsupported keys in 'vessel_widths.pvbm_mask': {unsupported}"
        )

    return PVBMMaskWidthConfig(
        direction_lag_px=_coerce_positive_float(
            raw_pvbm_mask.get("direction_lag_px", 6.0),
            "vessel_widths.pvbm_mask.direction_lag_px",
        ),
        max_asymmetry_px=_coerce_non_negative_float(
            raw_pvbm_mask.get("max_asymmetry_px", 1.0),
            "vessel_widths.pvbm_mask.max_asymmetry_px",
        ),
        trace_step_px=_coerce_positive_float(
            raw_pvbm_mask.get("trace_step_px", 1.0),
            "vessel_widths.pvbm_mask.trace_step_px",
        ),
        boundary_adjust_px=_coerce_non_negative_float(
            raw_pvbm_mask.get("boundary_adjust_px", 0.5),
            "vessel_widths.pvbm_mask.boundary_adjust_px",
        ),
        trace_padding_px=_coerce_non_negative_float(
            raw_pvbm_mask.get("trace_padding_px", 2.0),
            "vessel_widths.pvbm_mask.trace_padding_px",
        ),
    )


def _build_mask_width_config(
    raw_mask: Mapping[str, object],
) -> MaskWidthConfig:
    unsupported_keys = set(raw_mask) - {
        "tangent_window_px",
        "measurement_step_px",
        "boundary_refinement_steps",
        "trace_padding_px",
    }
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise ValueError(f"Unsupported keys in 'vessel_widths.mask': {unsupported}")

    return MaskWidthConfig(
        tangent_window_px=_coerce_positive_float(
            raw_mask.get("tangent_window_px", 10.0),
            "vessel_widths.mask.tangent_window_px",
        ),
        measurement_step_px=_coerce_positive_float(
            raw_mask.get("measurement_step_px", 0.25),
            "vessel_widths.mask.measurement_step_px",
        ),
        boundary_refinement_steps=_coerce_positive_int(
            raw_mask.get("boundary_refinement_steps", 12),
            "vessel_widths.mask.boundary_refinement_steps",
        ),
        trace_padding_px=_coerce_non_negative_float(
            raw_mask.get("trace_padding_px", 2.0),
            "vessel_widths.mask.trace_padding_px",
        ),
    )


def _build_profile_width_config(
    raw_profile: Mapping[str, object],
) -> ProfileWidthConfig:
    unsupported_keys = set(raw_profile) - {
        "image_source",
        "channel",
        "half_length_px",
        "sample_step_px",
        "smoothing_sigma_px",
        "boundary_method",
        "threshold_alpha",
        "min_contrast",
        "min_width_px",
        "max_width_px",
        "use_mask_guardrail",
        "mask_guardrail_min_ratio",
        "mask_guardrail_max_ratio",
        "fallback_to_mask",
    }
    if unsupported_keys:
        unsupported = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise ValueError(f"Unsupported keys in 'vessel_widths.profile': {unsupported}")

    min_width_px = _coerce_positive_float(
        raw_profile.get("min_width_px", 1.0),
        "vessel_widths.profile.min_width_px",
    )
    max_width_px = _coerce_positive_float(
        raw_profile.get("max_width_px", 80.0),
        "vessel_widths.profile.max_width_px",
    )
    if max_width_px < min_width_px:
        raise ValueError(
            "'vessel_widths.profile.max_width_px' must be greater than or equal to "
            "'vessel_widths.profile.min_width_px'"
        )

    min_ratio = _coerce_positive_float(
        raw_profile.get("mask_guardrail_min_ratio", 0.4),
        "vessel_widths.profile.mask_guardrail_min_ratio",
    )
    max_ratio = _coerce_positive_float(
        raw_profile.get("mask_guardrail_max_ratio", 2.5),
        "vessel_widths.profile.mask_guardrail_max_ratio",
    )
    if max_ratio < min_ratio:
        raise ValueError(
            "'vessel_widths.profile.mask_guardrail_max_ratio' must be greater than or "
            "equal to 'vessel_widths.profile.mask_guardrail_min_ratio'"
        )

    return ProfileWidthConfig(
        image_source=_coerce_required_string(
            raw_profile.get("image_source", "preprocessed_rgb"),
            "vessel_widths.profile.image_source",
        ),
        channel=_coerce_choice(
            raw_profile.get("channel", "green"),
            "vessel_widths.profile.channel",
            {"red", "green", "blue"},
        ),
        half_length_px=_coerce_positive_float(
            raw_profile.get("half_length_px", 20.0),
            "vessel_widths.profile.half_length_px",
        ),
        sample_step_px=_coerce_positive_float(
            raw_profile.get("sample_step_px", 0.25),
            "vessel_widths.profile.sample_step_px",
        ),
        smoothing_sigma_px=_coerce_non_negative_float(
            raw_profile.get("smoothing_sigma_px", 1.0),
            "vessel_widths.profile.smoothing_sigma_px",
        ),
        boundary_method=_coerce_choice(
            raw_profile.get("boundary_method", "half_depth"),
            "vessel_widths.profile.boundary_method",
            {"half_depth"},
        ),
        threshold_alpha=_coerce_float_in_range(
            raw_profile.get("threshold_alpha", 0.5),
            "vessel_widths.profile.threshold_alpha",
            minimum=0.0,
            maximum=1.0,
        ),
        min_contrast=_coerce_non_negative_float(
            raw_profile.get("min_contrast", 0.05),
            "vessel_widths.profile.min_contrast",
        ),
        min_width_px=min_width_px,
        max_width_px=max_width_px,
        use_mask_guardrail=_coerce_bool(
            raw_profile.get("use_mask_guardrail", True),
            "vessel_widths.profile.use_mask_guardrail",
        ),
        mask_guardrail_min_ratio=min_ratio,
        mask_guardrail_max_ratio=max_ratio,
        fallback_to_mask=_coerce_bool(
            raw_profile.get("fallback_to_mask", False),
            "vessel_widths.profile.fallback_to_mask",
        ),
    )


def _coerce_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{field_name}' must be a boolean")


def _coerce_positive_float(value: object, field_name: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    raise ValueError(f"'{field_name}' must be a positive number")


def _coerce_non_negative_float(value: object, field_name: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    raise ValueError(f"'{field_name}' must be a non-negative number")


def _coerce_float_in_range(
    value: object,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        if minimum <= numeric_value <= maximum:
            return numeric_value
    raise ValueError(f"'{field_name}' must be a number between {minimum} and {maximum}")


def _coerce_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError(f"'{field_name}' must be a positive integer")


def _coerce_nonzero_int(value: object, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value != 0:
        return value
    raise ValueError(f"'{field_name}' must be a non-zero integer")


def _coerce_optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"'{field_name}' must be a non-empty string or null")


def _coerce_required_string(value: object, field_name: str) -> str:
    normalized = _coerce_optional_string(value, field_name)
    if normalized is None:
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return normalized


def _coerce_choice(
    value: object,
    field_name: str,
    allowed_values: set[str],
) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in allowed_values:
            return normalized
    allowed = ", ".join(sorted(allowed_values))
    raise ValueError(f"'{field_name}' must be one of: {allowed}")


def _coerce_circle_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string")
    normalized = value.strip()
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"'{field_name}' must not contain path separators")
    return normalized


def _parse_rgb(value: object, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, str):
        return _parse_hex_color(value, field_name)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        channels = tuple(value)
        if len(channels) != 3:
            raise ValueError(f"'{field_name}' must contain exactly 3 channels")
        return tuple(_coerce_channel(channel, field_name) for channel in channels)
    raise ValueError(
        f"'{field_name}' must be a '#RRGGBB' string or a 3-item RGB sequence"
    )


def _parse_hex_color(value: str, field_name: str) -> tuple[int, int, int]:
    normalized = value.strip()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    if len(normalized) != 6:
        raise ValueError(f"'{field_name}' must be a 6-digit hex color")
    try:
        return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"'{field_name}' must be a valid hex color") from exc


def _coerce_channel(value: object, field_name: str) -> int:
    if isinstance(value, int) and 0 <= value <= 255:
        return value
    raise ValueError(f"'{field_name}' channels must be integers between 0 and 255")
