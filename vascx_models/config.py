from __future__ import annotations

from pathlib import Path

import yaml

from .config_schema import parse_app_config
from .config_types import (
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
    default_overlay_circles,
)

DEFAULT_CONFIG_NAME = "config.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


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

    return parse_app_config(raw_config, source_path=resolved_path)


__all__ = [
    "AppConfig",
    "DEFAULT_CONFIG_NAME",
    "MaskWidthConfig",
    "OverlayCircle",
    "OverlayColors",
    "OverlayConfig",
    "OverlayLayers",
    "PVBMMaskWidthConfig",
    "ProfileWidthConfig",
    "VesselTortuosityConfig",
    "VesselWidthConfig",
    "default_config_candidates",
    "default_overlay_circles",
    "load_app_config",
    "resolve_config_path",
]
