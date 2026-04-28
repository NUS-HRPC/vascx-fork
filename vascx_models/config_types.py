from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_DERIVED_CIRCLE_COLOR = (0, 255, 0)


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
class VesselWidthConfig:
    enabled: bool = True
    inner_circle: str | None = "2r"
    outer_circle: str | None = "3r"
    samples_per_connection: int = 5
    boundary_tolerance_px: float = 1.5
    method: str = "mask"
    mask: MaskWidthConfig = field(default_factory=MaskWidthConfig)
    pvbm_mask: PVBMMaskWidthConfig = field(default_factory=PVBMMaskWidthConfig)
    profile: ProfileWidthConfig = field(default_factory=ProfileWidthConfig)


@dataclass(frozen=True)
class VesselTortuosityConfig:
    enabled: bool = True
    inner_circle: str | None = "2r"
    outer_circle: str | None = "5r"


@dataclass(frozen=True)
class AppConfig:
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    vessel_widths: VesselWidthConfig = field(default_factory=VesselWidthConfig)
    vessel_tortuosities: VesselTortuosityConfig = field(
        default_factory=VesselTortuosityConfig
    )
    source_path: Path | None = None

