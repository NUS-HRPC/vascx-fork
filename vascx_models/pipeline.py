import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import click
import pandas as pd

logger = logging.getLogger(__name__)

VESSEL_METRIC_INTERMEDIATE_DIRS = ("vessels", "artery_vein")
VESSEL_METRIC_INTERMEDIATE_FILES = ("disc_geometry.csv",)
VESSEL_METRIC_OPTIONAL_BASE_DIRS = ("disc", "preprocessed_rgb")
VESSEL_METRIC_OPTIONAL_BASE_FILES = ("fovea.csv",)


@dataclass(frozen=True)
class PipelineDependencies:
    available_device_types: Callable[..., object]
    resolve_device: Callable[..., object]
    run_quality_estimation: Callable[..., object]
    run_fovea_detection: Callable[..., object]
    run_segmentation_vessels_and_av: Callable[..., object]
    run_segmentation_disc: Callable[..., object]
    generate_disc_circles: Callable[..., object]
    resolve_vessel_width_circle_pair: Callable[..., object]
    measure_vessel_widths_between_disc_circle_pair: Callable[..., object]
    compute_revised_crx_from_widths: Callable[..., object]
    measure_vessel_tortuosities_between_disc_circle_pair: Callable[..., object]
    summarize_vessel_tortuosities: Callable[..., object]
    measure_vessel_branching_between_disc_circle_pair: Callable[..., object]
    select_vessel_width_measurements_for_equivalents: Callable[..., object]
    batch_create_overlays: Callable[..., object]
    run_preprocessing: Callable[..., object]


def ensure_empty_or_new_output_dir(output_path: Path) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        raise click.ClickException(
            f"Output path already exists and is not empty: {output_path}"
        )
    output_path.mkdir(exist_ok=True, parents=True)


def copy_pipeline_output_for_vessel_metrics(
    source_output_path: Path,
    output_path: Path,
) -> tuple[Path, Path, Path]:
    missing: list[str] = []
    for dirname in VESSEL_METRIC_INTERMEDIATE_DIRS:
        if not (source_output_path / dirname).is_dir():
            missing.append(dirname)
    for filename in VESSEL_METRIC_INTERMEDIATE_FILES:
        if not (source_output_path / filename).is_file():
            missing.append(filename)
    if missing:
        raise click.ClickException(
            "Missing required intermediate output(s): " + ", ".join(missing)
        )

    for dirname in (*VESSEL_METRIC_INTERMEDIATE_DIRS, *VESSEL_METRIC_OPTIONAL_BASE_DIRS):
        source = source_output_path / dirname
        if source.is_dir():
            shutil.copytree(source, output_path / dirname)

    for filename in (
        *VESSEL_METRIC_INTERMEDIATE_FILES,
        *VESSEL_METRIC_OPTIONAL_BASE_FILES,
    ):
        source = source_output_path / filename
        if source.is_file():
            shutil.copy2(source, output_path / filename)

    return (
        output_path / "vessels",
        output_path / "artery_vein",
        output_path / "disc_geometry.csv",
    )


def resolve_vessel_width_rgb_dir(output_path: Path, image_source: str) -> Path:
    candidate = Path(image_source).expanduser()
    if candidate.is_absolute():
        return candidate
    return output_path / candidate


def load_fovea_overlay_data(fovea_path: Path) -> dict[str, tuple[int, int]] | None:
    if not fovea_path.exists():
        return None

    df_fovea = pd.read_csv(fovea_path, index_col=0)
    if df_fovea.empty:
        return None

    return {
        str(image_id): (int(row["x_fovea"]), int(row["y_fovea"]))
        for image_id, row in df_fovea.iterrows()
    }


def remove_metric_outputs(
    output_path: Path,
    width_enabled: bool,
    tortuosity_enabled: bool,
    branching_enabled: bool,
) -> None:
    if not width_enabled:
        width_paths = (
            output_path / "vessel_widths.csv",
            output_path / "vessel_widths_summary.csv",
            output_path / "vessel_equivalents.csv",
            output_path / "vessel_width_overlays",
        )
        for path in width_paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    if not tortuosity_enabled:
        tort_paths = (
            output_path / "vessel_tortuosities.csv",
            output_path / "vessel_tortuosity_summary.csv",
            output_path / "vessel_tortuosity_overlays",
        )
        for path in tort_paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    if not branching_enabled:
        branch_paths = (
            output_path / "vessel_branching.csv",
            output_path / "vessel_branching_widths.csv",
            output_path / "vessel_branching_overlays",
        )
        for path in branch_paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def overlay_circle_dirs(output_path: Path, circles) -> dict[str, Path]:
    disc_circles_dir = output_path / "disc_circles"
    return {
        circle.name: disc_circles_dir / circle.name
        for circle in circles
        if (disc_circles_dir / circle.name).is_dir()
    }


def overlay_config_with_selected_circles(overlay_config, circle_names):
    selected_circle_names = set(circle_names)
    return replace(
        overlay_config,
        circles=tuple(
            circle
            for circle in overlay_config.circles
            if circle.name in selected_circle_names
        ),
    )


def tortuosity_overlay_config(overlay_config):
    return replace(
        overlay_config,
        layers=replace(
            overlay_config.layers,
            arteries=True,
            veins=True,
            disc=True,
            fovea=True,
            vessel_widths=False,
        ),
        colors=replace(overlay_config.colors, vessel=(0, 255, 0)),
    )


def width_overlay_config(overlay_config):
    return replace(
        overlay_config,
        layers=replace(overlay_config.layers, vessel_widths=True),
    )


def branching_overlay_config(overlay_config):
    return replace(
        overlay_config,
        layers=replace(
            overlay_config.layers,
            vessel_branching=True,
            vessel_widths=False,
        ),
    )


def metric_circle_names(config_section) -> tuple[str, str] | None:
    if (
        not config_section.enabled
        or config_section.inner_circle is None
        or config_section.outer_circle is None
    ):
        return None
    return (config_section.inner_circle, config_section.outer_circle)


def render_metric_overlays(
    output_path: Path,
    rgb_dir: Path,
    overlay_config,
    width_circle_names: tuple[str, str] | None,
    tortuosity_circle_names: tuple[str, str] | None,
    branching_circle_names: tuple[str, str] | None,
    av_dir: Path | None,
    disc_dir: Path | None,
    vessels_dir: Path,
    df_vessel_widths: pd.DataFrame | None,
    df_vessel_tortuosities: pd.DataFrame | None,
    df_vessel_branching: pd.DataFrame | None,
    df_vessel_branching_widths: pd.DataFrame | None,
    df_selected_equivalent_widths: pd.DataFrame | None,
    fovea_data: dict[str, tuple[int, int]] | None,
    deps: PipelineDependencies,
) -> None:
    circle_dirs = overlay_circle_dirs(output_path, overlay_config.circles)
    width_config = (
        width_overlay_config(
            overlay_config_with_selected_circles(overlay_config, width_circle_names)
        )
        if width_circle_names is not None
        else width_overlay_config(overlay_config)
    )
    width_circle_dirs = overlay_circle_dirs(output_path, width_config.circles)
    tort_config = (
        tortuosity_overlay_config(
            overlay_config_with_selected_circles(
                overlay_config,
                tortuosity_circle_names,
            )
        )
        if tortuosity_circle_names is not None
        else tortuosity_overlay_config(overlay_config)
    )
    tort_circle_dirs = overlay_circle_dirs(output_path, tort_config.circles)
    branch_config = (
        branching_overlay_config(
            overlay_config_with_selected_circles(overlay_config, branching_circle_names)
        )
        if branching_circle_names is not None
        else branching_overlay_config(overlay_config)
    )
    branch_circle_dirs = overlay_circle_dirs(output_path, branch_config.circles)

    deps.batch_create_overlays(
        rgb_dir=rgb_dir,
        output_dir=output_path / "overlays",
        av_dir=av_dir,
        disc_dir=disc_dir,
        vessels_dir=vessels_dir,
        circle_dirs=circle_dirs,
        vessel_width_data=df_vessel_widths,
        fovea_data=fovea_data,
        overlay_config=overlay_config,
    )

    if df_vessel_tortuosities is not None:
        deps.batch_create_overlays(
            rgb_dir=rgb_dir,
            output_dir=output_path / "vessel_tortuosity_overlays",
            av_dir=av_dir,
            disc_dir=disc_dir,
            vessels_dir=vessels_dir,
            circle_dirs=tort_circle_dirs,
            tortuosity_data=df_vessel_tortuosities,
            fovea_data=fovea_data,
            overlay_config=tort_config,
        )
        logger.info(
            "Vessel tortuosity overlays saved to %s",
            output_path / "vessel_tortuosity_overlays",
        )

    if df_selected_equivalent_widths is not None:
        deps.batch_create_overlays(
            rgb_dir=rgb_dir,
            output_dir=output_path / "vessel_width_overlays",
            av_dir=av_dir,
            disc_dir=disc_dir,
            vessels_dir=vessels_dir,
            circle_dirs=width_circle_dirs,
            vessel_width_data=df_selected_equivalent_widths,
            fovea_data=fovea_data,
            overlay_config=width_config,
        )
        logger.info(
            "Vessel width overlays saved to %s",
            output_path / "vessel_width_overlays",
        )

    if df_vessel_branching is not None:
        deps.batch_create_overlays(
            rgb_dir=rgb_dir,
            output_dir=output_path / "vessel_branching_overlays",
            av_dir=av_dir,
            disc_dir=disc_dir,
            vessels_dir=vessels_dir,
            circle_dirs=branch_circle_dirs,
            branching_data=df_vessel_branching,
            branching_width_data=df_vessel_branching_widths,
            fovea_data=fovea_data,
            overlay_config=branch_config,
        )
        logger.info(
            "Vessel branching overlays saved to %s",
            output_path / "vessel_branching_overlays",
        )


def refresh_vessel_metric_disc_artifacts(
    output_path: Path,
    app_config,
    deps: PipelineDependencies,
) -> None:
    disc_circles_dir = output_path / "disc_circles"
    if disc_circles_dir.exists():
        shutil.rmtree(disc_circles_dir)

    disc_dir = output_path / "disc"
    if not disc_dir.is_dir():
        logger.warning(
            "Skipping disc circle regeneration because %s is missing",
            disc_dir,
        )
        return

    if not app_config.overlay.circles:
        logger.info(
            "Skipping disc circle regeneration because no enabled metric uses circles"
        )
        return

    deps.generate_disc_circles(
        disc_dir=disc_dir,
        circle_output_dir=disc_circles_dir,
        circles=app_config.overlay.circles,
        measurements_path=output_path / "disc_geometry.csv",
    )
    logger.info("Disc circle regeneration complete in %s", output_path)


def refresh_vessel_metric_overlays(
    output_path: Path,
    app_config,
    df_vessel_widths: pd.DataFrame,
    df_vessel_tortuosities: pd.DataFrame,
    df_vessel_branching: pd.DataFrame,
    df_vessel_branching_widths: pd.DataFrame,
    df_connection_widths: pd.DataFrame,
    deps: PipelineDependencies,
) -> None:
    overlay_path = output_path / "overlays"
    vessel_width_overlay_path = output_path / "vessel_width_overlays"
    vessel_tortuosity_overlay_path = output_path / "vessel_tortuosity_overlays"
    vessel_branching_overlay_path = output_path / "vessel_branching_overlays"
    for overlay_dir in (
        overlay_path,
        vessel_width_overlay_path,
        vessel_tortuosity_overlay_path,
        vessel_branching_overlay_path,
    ):
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir)

    if not app_config.overlay.enabled:
        logger.info("Skipping overlay regeneration because overlays are disabled")
        return

    rgb_dir = output_path / "preprocessed_rgb"
    if not rgb_dir.is_dir():
        logger.warning(
            "Skipping overlay regeneration because %s is missing",
            rgb_dir,
        )
        return

    disc_dir = output_path / "disc"
    fovea_data = load_fovea_overlay_data(output_path / "fovea.csv")

    df_selected_equivalent_widths = None
    if df_vessel_widths is not None and df_connection_widths is not None:
        df_selected_equivalent_widths = (
            deps.select_vessel_width_measurements_for_equivalents(
                df_vessel_widths,
                df_connection_widths,
            )
        )
    render_metric_overlays(
        output_path=output_path,
        rgb_dir=rgb_dir,
        overlay_config=app_config.overlay,
        width_circle_names=metric_circle_names(app_config.vessel_widths),
        tortuosity_circle_names=metric_circle_names(app_config.vessel_tortuosities),
        branching_circle_names=metric_circle_names(app_config.vessel_branching),
        av_dir=output_path / "artery_vein",
        disc_dir=disc_dir if disc_dir.is_dir() else None,
        vessels_dir=output_path / "vessels",
        df_vessel_widths=df_vessel_widths,
        df_vessel_tortuosities=df_vessel_tortuosities,
        df_vessel_branching=df_vessel_branching,
        df_vessel_branching_widths=df_vessel_branching_widths,
        df_selected_equivalent_widths=df_selected_equivalent_widths,
        fovea_data=fovea_data,
        deps=deps,
    )
    logger.info("Overlay regeneration complete in %s", output_path)


def compute_and_save_vessel_metrics(
    vessels_path: Path,
    av_path: Path,
    disc_geometry_path: Path,
    output_path: Path,
    app_config,
    deps: PipelineDependencies,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    width_inner_circle = width_outer_circle = None
    tortuosity_inner_circle = tortuosity_outer_circle = None
    branching_inner_circle = branching_outer_circle = None
    if app_config.vessel_widths.enabled:
        width_inner_circle, width_outer_circle = deps.resolve_vessel_width_circle_pair(
            app_config.overlay.circles,
            inner_circle_name=app_config.vessel_widths.inner_circle,
            outer_circle_name=app_config.vessel_widths.outer_circle,
        )
    if app_config.vessel_tortuosities.enabled:
        tortuosity_inner_circle, tortuosity_outer_circle = (
            deps.resolve_vessel_width_circle_pair(
                app_config.overlay.circles,
                inner_circle_name=app_config.vessel_tortuosities.inner_circle,
                outer_circle_name=app_config.vessel_tortuosities.outer_circle,
            )
        )
    if app_config.vessel_branching.enabled:
        branching_inner_circle, branching_outer_circle = (
            deps.resolve_vessel_width_circle_pair(
                app_config.overlay.circles,
                inner_circle_name=app_config.vessel_branching.inner_circle,
                outer_circle_name=app_config.vessel_branching.outer_circle,
            )
        )

    remove_metric_outputs(
        output_path,
        width_enabled=app_config.vessel_widths.enabled,
        tortuosity_enabled=app_config.vessel_tortuosities.enabled,
        branching_enabled=app_config.vessel_branching.enabled,
    )

    if (
        not app_config.vessel_widths.enabled
        and not app_config.vessel_tortuosities.enabled
        and not app_config.vessel_branching.enabled
    ):
        logger.info(
            "Skipping vessel metrics because all vessel metric sections are disabled"
        )
        return (None, None, None, None, None, None, None)

    rgb_dir = None
    if (
        app_config.vessel_widths.enabled
        and app_config.vessel_widths.method == "profile"
    ):
        rgb_dir = resolve_vessel_width_rgb_dir(
            output_path,
            app_config.vessel_widths.profile.image_source,
        )

    vessel_widths_path = output_path / "vessel_widths.csv"
    vessel_tortuosities_path = output_path / "vessel_tortuosities.csv"
    vessel_tortuosity_summary_path = output_path / "vessel_tortuosity_summary.csv"
    vessel_widths_summary_path = output_path / "vessel_widths_summary.csv"
    vessel_branching_path = output_path / "vessel_branching.csv"
    vessel_branching_widths_path = output_path / "vessel_branching_widths.csv"

    df_vessel_widths = None
    df_connection_widths = None
    df_vessel_equivalents = None
    df_vessel_tortuosities = None
    df_vessel_tortuosity_summary = None
    df_vessel_branching = None
    df_vessel_branching_widths = None
    try:
        if app_config.vessel_widths.enabled:
            logger.info(
                "Measuring vessel widths between %s and %s with %d samples per connection using %s",
                width_inner_circle.name,
                width_outer_circle.name,
                app_config.vessel_widths.samples_per_connection,
                app_config.vessel_widths.method,
            )
            df_vessel_widths = deps.measure_vessel_widths_between_disc_circle_pair(
                vessels_dir=vessels_path,
                av_dir=av_path,
                disc_geometry_path=disc_geometry_path,
                inner_circle=width_inner_circle,
                outer_circle=width_outer_circle,
                output_path=vessel_widths_path,
                samples_per_connection=app_config.vessel_widths.samples_per_connection,
                boundary_tolerance_px=app_config.vessel_widths.boundary_tolerance_px,
                tangent_window_px=app_config.vessel_widths.mask.tangent_window_px,
                measurement_step_px=app_config.vessel_widths.mask.measurement_step_px,
                boundary_refinement_steps=app_config.vessel_widths.mask.boundary_refinement_steps,
                trace_padding_px=app_config.vessel_widths.mask.trace_padding_px,
                width_config=app_config.vessel_widths,
                rgb_dir=rgb_dir,
            )
            df_connection_widths, df_vessel_equivalents = (
                deps.compute_revised_crx_from_widths(df_vessel_widths)
            )
            legacy_equivalents_path = output_path / "vessel_equivalents.csv"
            if legacy_equivalents_path.exists():
                legacy_equivalents_path.unlink()
            df_vessel_equivalents.to_csv(vessel_widths_summary_path, index=False)
            logger.info("Vessel widths summary saved to %s", vessel_widths_summary_path)
        else:
            logger.info(
                "Skipping vessel width metrics because vessel_widths.enabled is false"
            )

        if app_config.vessel_tortuosities.enabled:
            logger.info(
                "Measuring vessel tortuosities between %s and %s",
                tortuosity_inner_circle.name,
                tortuosity_outer_circle.name,
            )
            df_vessel_tortuosities = (
                deps.measure_vessel_tortuosities_between_disc_circle_pair(
                    vessels_dir=vessels_path,
                    av_dir=av_path,
                    disc_geometry_path=disc_geometry_path,
                    inner_circle=tortuosity_inner_circle,
                    outer_circle=tortuosity_outer_circle,
                    output_path=vessel_tortuosities_path,
                )
            )
            df_vessel_tortuosity_summary = deps.summarize_vessel_tortuosities(
                df_vessel_tortuosities,
                output_path=vessel_tortuosity_summary_path,
            )
            logger.info(
                "Vessel tortuosity summary saved to %s",
                vessel_tortuosity_summary_path,
            )
        else:
            logger.info(
                "Skipping vessel tortuosity metrics because vessel_tortuosities.enabled is false"
            )

        if app_config.vessel_branching.enabled:
            logger.info(
                "Measuring vessel branching between %s and %s",
                branching_inner_circle.name,
                branching_outer_circle.name,
            )
            df_vessel_branching, df_vessel_branching_widths = (
                deps.measure_vessel_branching_between_disc_circle_pair(
                    vessels_dir=vessels_path,
                    av_dir=av_path,
                    disc_geometry_path=disc_geometry_path,
                    inner_circle=branching_inner_circle,
                    outer_circle=branching_outer_circle,
                    output_path=vessel_branching_path,
                    widths_output_path=vessel_branching_widths_path,
                    branching_config=app_config.vessel_branching,
                )
            )
            logger.info(
                "Vessel branching outputs saved to %s and %s",
                vessel_branching_path,
                vessel_branching_widths_path,
            )
        else:
            logger.info(
                "Skipping vessel branching metrics because vessel_branching.enabled is false"
            )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    return (
        df_vessel_widths,
        df_vessel_tortuosities,
        df_vessel_tortuosity_summary,
        df_connection_widths,
        df_vessel_equivalents,
        df_vessel_branching,
        df_vessel_branching_widths,
    )


def run_vessel_metrics_pipeline(
    source_output_path: Path,
    output_path: Path | None,
    app_config,
    deps: PipelineDependencies,
    default_timestamped_output_path,
) -> None:
    source_output_path = source_output_path.resolve()
    if output_path is None:
        output_path = default_timestamped_output_path().resolve()
    else:
        output_path = output_path.resolve()
    if source_output_path == output_path:
        raise click.ClickException(
            "OUTPUT_PATH must be different from SOURCE_OUTPUT_PATH"
        )
    try:
        output_path.relative_to(source_output_path)
    except ValueError:
        pass
    else:
        raise click.ClickException("OUTPUT_PATH must not be inside SOURCE_OUTPUT_PATH")

    ensure_empty_or_new_output_dir(output_path)
    vessels_path, av_path, disc_geometry_path = copy_pipeline_output_for_vessel_metrics(
        source_output_path=source_output_path,
        output_path=output_path,
    )
    refresh_vessel_metric_disc_artifacts(
        output_path=output_path,
        app_config=app_config,
        deps=deps,
    )
    (
        df_vessel_widths,
        df_vessel_tortuosities,
        _,
        df_connection_widths,
        _,
        df_vessel_branching,
        df_vessel_branching_widths,
    ) = compute_and_save_vessel_metrics(
        vessels_path=vessels_path,
        av_path=av_path,
        disc_geometry_path=disc_geometry_path,
        output_path=output_path,
        app_config=app_config,
        deps=deps,
    )
    refresh_vessel_metric_overlays(
        output_path=output_path,
        app_config=app_config,
        df_vessel_widths=df_vessel_widths,
        df_vessel_tortuosities=df_vessel_tortuosities,
        df_vessel_branching=df_vessel_branching,
        df_vessel_branching_widths=df_vessel_branching_widths,
        df_connection_widths=df_connection_widths,
        deps=deps,
    )
    logger.info("Vessel metrics complete. Results saved to %s", output_path)


def run_pipeline(
    data_path,
    output_path,
    app_config,
    deps: PipelineDependencies,
    preprocess,
    vessels,
    disc,
    quality,
    fovea,
    overlay,
    device_name,
    n_jobs,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    overlay_enabled = app_config.overlay.enabled if overlay is None else overlay
    if app_config.source_path is not None:
        logger.info("Loaded config from %s", app_config.source_path)

    preprocess_rgb_path = output_path / "preprocessed_rgb"
    vessels_path = output_path / "vessels"
    av_path = output_path / "artery_vein"
    disc_path = output_path / "disc"
    disc_circles_path = output_path / "disc_circles"
    overlay_path = output_path / "overlays"

    if preprocess:
        preprocess_rgb_path.mkdir(exist_ok=True, parents=True)
    if vessels:
        av_path.mkdir(exist_ok=True, parents=True)
        vessels_path.mkdir(exist_ok=True, parents=True)
    if disc:
        disc_path.mkdir(exist_ok=True, parents=True)
        if app_config.overlay.circles:
            disc_circles_path.mkdir(exist_ok=True, parents=True)
    if overlay_enabled:
        overlay_path.mkdir(exist_ok=True, parents=True)

    bounds_path = output_path / "bounds.csv" if preprocess else None
    quality_path = output_path / "quality.csv" if quality else None
    fovea_path = output_path / "fovea.csv" if fovea else None
    disc_geometry_path = output_path / "disc_geometry.csv" if disc else None
    vessel_widths_path = output_path / "vessel_widths.csv" if disc and vessels else None
    vessel_tortuosities_path = (
        output_path / "vessel_tortuosities.csv" if disc and vessels else None
    )
    vessel_tortuosity_summary_path = (
        output_path / "vessel_tortuosity_summary.csv" if disc and vessels else None
    )

    data_path = Path(data_path)
    is_csv = data_path.suffix.lower() == ".csv"

    files = []
    ids = None

    if is_csv:
        logger.info("Reading file paths from CSV: %s", data_path)
        try:
            df = pd.read_csv(data_path)
            if "path" not in df.columns:
                logger.error("CSV must contain a 'path' column")
                return

            files = [Path(path_value) for path_value in df["path"]]

            if "id" in df.columns:
                ids = df["id"].tolist()
                logger.info("Using IDs from CSV 'id' column")

        except Exception as exc:
            logger.exception("Error reading CSV file: %s", exc)
            return
    else:
        logger.info("Finding files in directory: %s", data_path)
        files = list(data_path.glob("*"))
        ids = [file_path.stem for file_path in files]

    if not files:
        logger.warning("No files found to process")
        return

    logger.info("Found %d files to process", len(files))

    if preprocess:
        logger.info("Running preprocessing")
        deps.run_preprocessing(
            files=files,
            ids=ids,
            rgb_path=preprocess_rgb_path,
            bounds_path=bounds_path,
            n_jobs=n_jobs,
        )
        preprocessed_files = list(preprocess_rgb_path.glob("*.png"))
    else:
        preprocessed_files = files
    ids = [file_path.stem for file_path in preprocessed_files]
    logger.info("Prepared %d images for inference", len(preprocessed_files))

    available_devices = deps.available_device_types()
    logger.info(
        "Device availability: cuda=%s, mps=%s, cpu=%s",
        available_devices["cuda"],
        available_devices["mps"],
        available_devices["cpu"],
    )
    try:
        device = deps.resolve_device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if device_name == "auto":
        logger.info("Auto-selected device: %s", device)
    else:
        logger.info("Using requested device '%s': %s", device_name, device)

    if quality:
        logger.info("Running quality estimation")
        df_quality = deps.run_quality_estimation(
            fpaths=preprocessed_files, ids=ids, device=device
        )
        df_quality.to_csv(quality_path)
        logger.info("Quality results saved to %s", quality_path)

    if vessels:
        logger.info("Running vessels and AV segmentation")
        deps.run_segmentation_vessels_and_av(
            rgb_paths=preprocessed_files,
            ids=ids,
            av_path=av_path,
            vessels_path=vessels_path,
            artery_color=app_config.overlay.colors.artery,
            vein_color=app_config.overlay.colors.vein,
            vessel_color=app_config.overlay.colors.vessel,
            device=device,
        )
        logger.info("Vessel segmentation saved to %s", vessels_path)
        logger.info("AV segmentation saved to %s", av_path)

    if disc:
        logger.info("Running optic disc segmentation")
        deps.run_segmentation_disc(
            rgb_paths=preprocessed_files,
            ids=ids,
            output_path=disc_path,
            disc_color=app_config.overlay.colors.disc,
            device=device,
        )
        logger.info("Disc segmentation saved to %s", disc_path)
        if app_config.overlay.circles:
            deps.generate_disc_circles(
                disc_dir=disc_path,
                circle_output_dir=disc_circles_path,
                circles=app_config.overlay.circles,
                measurements_path=disc_geometry_path,
            )
            logger.info("Disc circles saved to %s", disc_circles_path)
        else:
            logger.info(
                "Skipping disc circle generation because vessel metrics are disabled"
            )

    df_vessel_widths = None
    df_vessel_tortuosities = None
    df_vessel_branching = None
    df_vessel_branching_widths = None
    df_selected_equivalent_widths = None
    if (
        disc
        and vessels
        and (
            app_config.vessel_widths.enabled
            or app_config.vessel_tortuosities.enabled
            or app_config.vessel_branching.enabled
        )
    ):
        (
            df_vessel_widths,
            df_vessel_tortuosities,
            _,
            df_connection_widths,
            _,
            df_vessel_branching,
            df_vessel_branching_widths,
        ) = compute_and_save_vessel_metrics(
            vessels_path=vessels_path,
            av_path=av_path,
            disc_geometry_path=disc_geometry_path,
            output_path=output_path,
            app_config=app_config,
            deps=deps,
        )
        if df_vessel_widths is not None and df_connection_widths is not None:
            df_selected_equivalent_widths = (
                deps.select_vessel_width_measurements_for_equivalents(
                    df_vessel_widths,
                    df_connection_widths,
                )
            )
        if df_vessel_tortuosities is not None:
            logger.info(
                "Vessel tortuosity outputs saved to %s and %s",
                vessel_tortuosities_path,
                vessel_tortuosity_summary_path,
            )
        if df_vessel_branching is not None:
            logger.info("Vessel branching outputs saved to %s", output_path)
    elif disc and vessels:
        logger.info(
            "Skipping vessel metrics because all vessel metric sections are disabled"
        )

    df_fovea = None
    if fovea:
        logger.info("Running fovea detection")
        df_fovea = deps.run_fovea_detection(
            rgb_paths=preprocessed_files, ids=ids, device=device
        )
        df_fovea.to_csv(fovea_path)
        logger.info("Fovea detection results saved to %s", fovea_path)

    if overlay_enabled:
        logger.info("Creating visualization overlays")

        fovea_data = None
        if df_fovea is not None:
            fovea_data = {
                idx: (row["x_fovea"], row["y_fovea"])
                for idx, row in df_fovea.iterrows()
            }
        render_metric_overlays(
            output_path=output_path,
            rgb_dir=preprocess_rgb_path if preprocess else data_path,
            overlay_config=app_config.overlay,
            width_circle_names=metric_circle_names(app_config.vessel_widths),
            tortuosity_circle_names=metric_circle_names(app_config.vessel_tortuosities),
            branching_circle_names=metric_circle_names(app_config.vessel_branching),
            av_dir=av_path,
            disc_dir=disc_path,
            vessels_dir=vessels_path,
            df_vessel_widths=df_vessel_widths,
            df_vessel_tortuosities=df_vessel_tortuosities if disc and vessels else None,
            df_vessel_branching=df_vessel_branching if disc and vessels else None,
            df_vessel_branching_widths=(
                df_vessel_branching_widths if disc and vessels else None
            ),
            df_selected_equivalent_widths=df_selected_equivalent_widths,
            fovea_data=fovea_data,
            deps=deps,
        )

        logger.info("Visualization overlays saved to %s", overlay_path)

    logger.info("All requested processing complete. Results saved to %s", output_path)
