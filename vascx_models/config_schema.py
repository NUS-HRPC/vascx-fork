from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config_types import (
    DEFAULT_DERIVED_CIRCLE_COLOR,
    AppConfig,
    MaskWidthConfig,
    OverlayCircle,
    OverlayColors,
    OverlayConfig,
    OverlayLayers,
    PVBMMaskWidthConfig,
    ProfileWidthConfig,
    VesselTortuosityConfig,
    VesselWidthConfig,
)


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _OverlayLayers(_ConfigModel):
    _aliases: ClassVar[dict[str, str]] = {
        "artery": "arteries",
        "arteries": "arteries",
        "vein": "veins",
        "veins": "veins",
        "disc": "disc",
        "fovea": "fovea",
    }

    arteries: bool = True
    veins: bool = True
    disc: bool = True
    fovea: bool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_layer_aliases(cls, data: Any) -> Any:
        data = _mapping_or_empty(data, "'overlay.layers' must be a mapping")
        normalized: dict[str, Any] = {}
        for raw_key, value in data.items():
            if raw_key not in cls._aliases:
                raise ValueError(f"Unsupported overlay layer '{raw_key}'")
            normalized[cls._aliases[raw_key]] = value
        return normalized

    @field_validator("arteries", "veins", "disc", "fovea", mode="before")
    @classmethod
    def validate_bool(cls, value: Any, info: Any) -> bool:
        return _strict_bool(value, f"overlay.layers.{info.field_name}")

    def to_config(self) -> OverlayLayers:
        return OverlayLayers(
            arteries=self.arteries,
            veins=self.veins,
            disc=self.disc,
            fovea=self.fovea,
            vessel_widths=False,
        )


class _OverlayColors(_ConfigModel):
    _aliases: ClassVar[dict[str, str]] = {
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

    artery: tuple[int, int, int] = (255, 0, 0)
    vein: tuple[int, int, int] = (0, 0, 255)
    vessel: tuple[int, int, int] = (255, 255, 255)
    disc: tuple[int, int, int] = (255, 255, 255)
    fovea: tuple[int, int, int] = (255, 255, 0)
    vessel_width: tuple[int, int, int] = (0, 0, 0)

    @model_validator(mode="before")
    @classmethod
    def normalize_color_aliases(cls, data: Any) -> Any:
        data = _mapping_or_empty(data, "'overlay.colors' must be a mapping")
        normalized: dict[str, Any] = {}
        for raw_key, value in data.items():
            if raw_key not in cls._aliases:
                raise ValueError(f"Unsupported overlay color '{raw_key}'")
            normalized[cls._aliases[raw_key]] = value
        return normalized

    @field_validator(
        "artery",
        "vein",
        "vessel",
        "disc",
        "fovea",
        "vessel_width",
        mode="before",
    )
    @classmethod
    def parse_color(cls, value: Any, info: Any) -> tuple[int, int, int]:
        return _parse_rgb(value, f"overlay.colors.{info.field_name}")

    def to_config(self) -> OverlayColors:
        return OverlayColors(
            artery=self.artery,
            vein=self.vein,
            vessel=self.vessel,
            disc=self.disc,
            fovea=self.fovea,
            vessel_width=self.vessel_width,
        )


class _OverlayConfig(_ConfigModel):
    enabled: bool = True
    layers: _OverlayLayers = Field(default_factory=_OverlayLayers)
    colors: _OverlayColors = Field(default_factory=_OverlayColors)
    circle_colors: dict[str, tuple[int, int, int]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_overlay(cls, data: Any) -> Any:
        data = _mapping_or_empty(data, "'overlay' must be a mapping")
        unsupported = set(data) - {
            "enabled",
            "layers",
            "colors",
            "colours",
            "circle_colors",
            "circle_colours",
        }
        if unsupported:
            names = ", ".join(sorted(str(key) for key in unsupported))
            raise ValueError(f"Unsupported keys in 'overlay': {names}")

        normalized = dict(data)
        if "colors" not in normalized and "colours" in normalized:
            normalized["colors"] = normalized.pop("colours")
        if "circle_colors" not in normalized and "circle_colours" in normalized:
            normalized["circle_colors"] = normalized.pop("circle_colours")
        normalized.pop("colours", None)
        normalized.pop("circle_colours", None)
        return normalized

    @field_validator("enabled", mode="before")
    @classmethod
    def validate_enabled(cls, value: Any) -> bool:
        return _strict_bool(value, "overlay.enabled")

    @field_validator("circle_colors", mode="before")
    @classmethod
    def parse_circle_colors(cls, value: Any) -> dict[str, tuple[int, int, int]]:
        value = _mapping_or_empty(value, "'overlay.circle_colors' must be a mapping")
        return {
            _coerce_circle_name(raw_key, "overlay.circle_colors"): _parse_rgb(
                raw_value,
                f"overlay.circle_colors.{raw_key}",
            )
            for raw_key, raw_value in value.items()
        }


class _PVBMMaskWidthConfig(_ConfigModel):
    direction_lag_px: float = 6.0
    max_asymmetry_px: float = 1.0
    trace_step_px: float = 1.0
    boundary_adjust_px: float = 0.5
    trace_padding_px: float = 2.0

    @model_validator(mode="before")
    @classmethod
    def validate_mapping(cls, data: Any) -> Any:
        return _mapping_or_empty(data, "'vessel_widths.pvbm_mask' must be a mapping")

    @field_validator("direction_lag_px", "trace_step_px", mode="before")
    @classmethod
    def validate_positive_float(cls, value: Any, info: Any) -> float:
        return _positive_float(value, f"vessel_widths.pvbm_mask.{info.field_name}")

    @field_validator(
        "max_asymmetry_px",
        "boundary_adjust_px",
        "trace_padding_px",
        mode="before",
    )
    @classmethod
    def validate_non_negative_float(cls, value: Any, info: Any) -> float:
        return _non_negative_float(
            value,
            f"vessel_widths.pvbm_mask.{info.field_name}",
        )

    def to_config(self) -> PVBMMaskWidthConfig:
        return PVBMMaskWidthConfig(**self.model_dump())


class _MaskWidthConfig(_ConfigModel):
    tangent_window_px: float = 10.0
    measurement_step_px: float = 0.25
    boundary_refinement_steps: int = 12
    trace_padding_px: float = 2.0

    @model_validator(mode="before")
    @classmethod
    def validate_mapping(cls, data: Any) -> Any:
        return _mapping_or_empty(data, "'vessel_widths.mask' must be a mapping")

    @field_validator("tangent_window_px", "measurement_step_px", mode="before")
    @classmethod
    def validate_positive_float(cls, value: Any, info: Any) -> float:
        return _positive_float(value, f"vessel_widths.mask.{info.field_name}")

    @field_validator("boundary_refinement_steps", mode="before")
    @classmethod
    def validate_positive_int(cls, value: Any) -> int:
        return _positive_int(value, "vessel_widths.mask.boundary_refinement_steps")

    @field_validator("trace_padding_px", mode="before")
    @classmethod
    def validate_non_negative_float(cls, value: Any) -> float:
        return _non_negative_float(value, "vessel_widths.mask.trace_padding_px")

    def to_config(self) -> MaskWidthConfig:
        return MaskWidthConfig(**self.model_dump())


class _ProfileWidthConfig(_ConfigModel):
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

    @model_validator(mode="before")
    @classmethod
    def validate_mapping(cls, data: Any) -> Any:
        return _mapping_or_empty(data, "'vessel_widths.profile' must be a mapping")

    @field_validator("image_source", mode="before")
    @classmethod
    def validate_image_source(cls, value: Any) -> str:
        return _required_string(value, "vessel_widths.profile.image_source")

    @field_validator("channel", mode="before")
    @classmethod
    def validate_channel(cls, value: Any) -> str:
        return _choice(value, "vessel_widths.profile.channel", {"red", "green", "blue"})

    @field_validator("boundary_method", mode="before")
    @classmethod
    def validate_boundary_method(cls, value: Any) -> str:
        return _choice(
            value,
            "vessel_widths.profile.boundary_method",
            {"half_depth"},
        )

    @field_validator(
        "half_length_px",
        "sample_step_px",
        "min_width_px",
        "max_width_px",
        "mask_guardrail_min_ratio",
        "mask_guardrail_max_ratio",
        mode="before",
    )
    @classmethod
    def validate_positive_float(cls, value: Any, info: Any) -> float:
        return _positive_float(value, f"vessel_widths.profile.{info.field_name}")

    @field_validator("smoothing_sigma_px", "min_contrast", mode="before")
    @classmethod
    def validate_non_negative_float(cls, value: Any, info: Any) -> float:
        return _non_negative_float(value, f"vessel_widths.profile.{info.field_name}")

    @field_validator("threshold_alpha", mode="before")
    @classmethod
    def validate_threshold_alpha(cls, value: Any) -> float:
        return _float_in_range(
            value,
            "vessel_widths.profile.threshold_alpha",
            minimum=0.0,
            maximum=1.0,
        )

    @field_validator("use_mask_guardrail", "fallback_to_mask", mode="before")
    @classmethod
    def validate_bool(cls, value: Any, info: Any) -> bool:
        return _strict_bool(value, f"vessel_widths.profile.{info.field_name}")

    @model_validator(mode="after")
    def validate_ordered_ranges(self) -> _ProfileWidthConfig:
        if self.max_width_px < self.min_width_px:
            raise ValueError(
                "'vessel_widths.profile.max_width_px' must be greater than or "
                "equal to 'vessel_widths.profile.min_width_px'"
            )
        if self.mask_guardrail_max_ratio < self.mask_guardrail_min_ratio:
            raise ValueError(
                "'vessel_widths.profile.mask_guardrail_max_ratio' must be greater "
                "than or equal to 'vessel_widths.profile.mask_guardrail_min_ratio'"
            )
        return self

    def to_config(self) -> ProfileWidthConfig:
        return ProfileWidthConfig(**self.model_dump())


class _VesselWidthConfig(_ConfigModel):
    enabled: bool = True
    inner_circle: str | None = "2r"
    outer_circle: str | None = "3r"
    samples_per_connection: int = 5
    boundary_tolerance_px: float = 1.5
    method: str = "mask"
    mask: _MaskWidthConfig = Field(default_factory=_MaskWidthConfig)
    pvbm_mask: _PVBMMaskWidthConfig = Field(default_factory=_PVBMMaskWidthConfig)
    profile: _ProfileWidthConfig = Field(default_factory=_ProfileWidthConfig)

    @model_validator(mode="before")
    @classmethod
    def validate_mapping(cls, data: Any) -> Any:
        return _mapping_or_empty(data, "'vessel_widths' must be a mapping")

    @field_validator("enabled", mode="before")
    @classmethod
    def validate_enabled(cls, value: Any) -> bool:
        return _strict_bool(value, "vessel_widths.enabled")

    @field_validator("inner_circle", "outer_circle", mode="before")
    @classmethod
    def validate_circle_name(cls, value: Any, info: Any) -> str | None:
        return _optional_string(value, f"vessel_widths.{info.field_name}")

    @field_validator("samples_per_connection", mode="before")
    @classmethod
    def validate_samples_per_connection(cls, value: Any) -> int:
        return _nonzero_int(value, "vessel_widths.samples_per_connection")

    @field_validator("boundary_tolerance_px", mode="before")
    @classmethod
    def validate_boundary_tolerance(cls, value: Any) -> float:
        return _positive_float(value, "vessel_widths.boundary_tolerance_px")

    @field_validator("method", mode="before")
    @classmethod
    def validate_method(cls, value: Any) -> str:
        return _choice(
            value,
            "vessel_widths.method",
            {"mask", "pvbm_mask", "profile"},
        )

    def to_config(self) -> VesselWidthConfig:
        return VesselWidthConfig(
            enabled=self.enabled,
            inner_circle=self.inner_circle,
            outer_circle=self.outer_circle,
            samples_per_connection=self.samples_per_connection,
            boundary_tolerance_px=self.boundary_tolerance_px,
            method=self.method,
            mask=self.mask.to_config(),
            pvbm_mask=self.pvbm_mask.to_config(),
            profile=self.profile.to_config(),
        )


class _VesselTortuosityConfig(_ConfigModel):
    enabled: bool = True
    inner_circle: str | None = "2r"
    outer_circle: str | None = "5r"

    @model_validator(mode="before")
    @classmethod
    def validate_mapping(cls, data: Any) -> Any:
        return _mapping_or_empty(data, "'vessel_tortuosities' must be a mapping")

    @field_validator("enabled", mode="before")
    @classmethod
    def validate_enabled(cls, value: Any) -> bool:
        return _strict_bool(value, "vessel_tortuosities.enabled")

    @field_validator("inner_circle", "outer_circle", mode="before")
    @classmethod
    def validate_circle_name(cls, value: Any, info: Any) -> str | None:
        return _optional_string(value, f"vessel_tortuosities.{info.field_name}")

    def to_config(self) -> VesselTortuosityConfig:
        return VesselTortuosityConfig(
            enabled=self.enabled,
            inner_circle=self.inner_circle,
            outer_circle=self.outer_circle,
        )


class _AppConfig(_ConfigModel):
    model_config = ConfigDict(extra="ignore")

    overlay: _OverlayConfig = Field(default_factory=_OverlayConfig)
    vessel_widths: _VesselWidthConfig = Field(default_factory=_VesselWidthConfig)
    vessel_tortuosities: _VesselTortuosityConfig = Field(
        default_factory=_VesselTortuosityConfig
    )

    @model_validator(mode="before")
    @classmethod
    def validate_mapping(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping")
        return data


def parse_app_config(raw_config: object, source_path: Path) -> AppConfig:
    validated = _AppConfig.model_validate(raw_config)
    vessel_widths = validated.vessel_widths.to_config()
    vessel_tortuosities = validated.vessel_tortuosities.to_config()
    circle_colors = validated.overlay.circle_colors
    circles = _derive_overlay_circles(
        vessel_widths,
        vessel_tortuosities,
        circle_color_overrides=circle_colors,
    )

    return AppConfig(
        overlay=OverlayConfig(
            enabled=validated.overlay.enabled,
            layers=validated.overlay.layers.to_config(),
            colors=validated.overlay.colors.to_config(),
            circles=circles,
        ),
        vessel_widths=vessel_widths,
        vessel_tortuosities=vessel_tortuosities,
        source_path=source_path,
    )


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
                    normalized_name,
                    DEFAULT_DERIVED_CIRCLE_COLOR,
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

    return tuple(
        sorted(
            circles_by_name.values(),
            key=lambda circle: (circle.diameter, circle.name),
        )
    )


def _mapping_or_empty(value: Any, error_message: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise ValueError(error_message)


def _strict_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{field_name}' must be a boolean")


def _positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    raise ValueError(f"'{field_name}' must be a positive number")


def _non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    raise ValueError(f"'{field_name}' must be a non-negative number")


def _float_in_range(
    value: Any,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        if minimum <= numeric_value <= maximum:
            return numeric_value
    raise ValueError(f"'{field_name}' must be a number between {minimum} and {maximum}")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError(f"'{field_name}' must be a positive integer")


def _nonzero_int(value: Any, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value != 0:
        return value
    raise ValueError(f"'{field_name}' must be a non-zero integer")


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"'{field_name}' must be a non-empty string or null")


def _required_string(value: Any, field_name: str) -> str:
    normalized = _optional_string(value, field_name)
    if normalized is None:
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return normalized


def _choice(value: Any, field_name: str, allowed_values: set[str]) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in allowed_values:
            return normalized
    allowed = ", ".join(sorted(allowed_values))
    raise ValueError(f"'{field_name}' must be one of: {allowed}")


def _coerce_circle_name(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string")
    normalized = value.strip()
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"'{field_name}' must not contain path separators")
    return normalized


def _circle_name_to_diameter(circle_name: str, field_name: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)r", circle_name)
    if match is None:
        raise ValueError(
            f"'{field_name}' must use the '<multiplier>r' format, for example "
            "'2r' or '3.5r'"
        )
    return _positive_float(float(match.group(1)), field_name)


def _parse_rgb(value: Any, field_name: str) -> tuple[int, int, int]:
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


def _coerce_channel(value: Any, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 255:
        return value
    raise ValueError(f"'{field_name}' channels must be integers between 0 and 255")
