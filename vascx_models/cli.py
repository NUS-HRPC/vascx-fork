import logging
import shutil
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from rtnls_fundusprep.cli import _run_preprocessing

from .config import load_app_config
from .disc_circles import generate_disc_circles
from .runtime import configure_runtime_environment

configure_runtime_environment()

from .inference import (
    available_device_types,
    resolve_device,
    run_fovea_detection,
    run_quality_estimation,
    run_segmentation_disc,
    run_segmentation_vessels_and_av,
)
from .utils import batch_create_overlays
from .vessel_tortuosities import measure_vessel_tortuosities_between_disc_circle_pair
from .vessel_widths import (
    compute_revised_crx_from_widths,
    measure_vessel_widths_between_disc_circle_pair,
    resolve_vessel_width_circle_pair,
    select_vessel_width_measurements_for_equivalents,
)

logger = logging.getLogger(__name__)

VESSEL_METRIC_INTERMEDIATE_DIRS = ("vessels", "artery_vein")
VESSEL_METRIC_INTERMEDIATE_FILES = ("disc_geometry.csv",)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    warnings.filterwarnings(
        "ignore",
        message=(
            "Using a non-tuple sequence for multidimensional indexing is deprecated "
            "and will be changed in pytorch 2.9; use x\\[tuple\\(seq\\)\\] instead of x\\[seq\\].*"
        ),
        category=UserWarning,
        module=r"monai\.inferers\.utils",
    )


def _ensure_empty_or_new_output_dir(output_path: Path) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        raise click.ClickException(
            f"Output path already exists and is not empty: {output_path}"
        )
    output_path.mkdir(exist_ok=True, parents=True)


def _default_timestamped_output_path(base_dir: Path | None = None) -> Path:
    parent = Path.cwd() if base_dir is None else base_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return parent / f"output_{timestamp}"


def _copy_pipeline_output_for_vessel_metrics(
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

    for item in source_output_path.iterdir():
        destination = output_path / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    return (
        output_path / "vessels",
        output_path / "artery_vein",
        output_path / "disc_geometry.csv",
    )


def _resolve_vessel_width_rgb_dir(output_path: Path, image_source: str) -> Path:
    candidate = Path(image_source).expanduser()
    if candidate.is_absolute():
        return candidate
    return output_path / candidate


def _load_fovea_overlay_data(fovea_path: Path) -> dict[str, tuple[int, int]] | None:
    if not fovea_path.exists():
        return None

    df_fovea = pd.read_csv(fovea_path, index_col=0)
    if df_fovea.empty:
        return None

    return {
        str(image_id): (int(row["x_fovea"]), int(row["y_fovea"]))
        for image_id, row in df_fovea.iterrows()
    }


def _refresh_vessel_metric_disc_artifacts(
    output_path: Path,
    config_path: Path | None = None,
) -> None:
    disc_circles_dir = output_path / "disc_circles"
    if disc_circles_dir.exists():
        shutil.rmtree(disc_circles_dir)

    try:
        app_config = load_app_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    disc_dir = output_path / "disc"
    if not disc_dir.is_dir():
        logger.warning(
            "Skipping disc circle regeneration because %s is missing",
            disc_dir,
        )
        return

    generate_disc_circles(
        disc_dir=disc_dir,
        circle_output_dir=disc_circles_dir,
        circles=app_config.overlay.circles,
        measurements_path=output_path / "disc_geometry.csv",
    )
    logger.info("Disc circle regeneration complete in %s", output_path)


def _refresh_vessel_metric_overlays(
    output_path: Path,
    df_vessel_widths: pd.DataFrame,
    df_connection_widths: pd.DataFrame,
    config_path: Path | None = None,
) -> None:
    overlay_path = output_path / "overlays"
    vessel_equivalent_overlay_path = output_path / "vessel_equivalent_overlays"
    for overlay_dir in (overlay_path, vessel_equivalent_overlay_path):
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir)

    try:
        app_config = load_app_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

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
    disc_circles_dir = output_path / "disc_circles"
    circle_dirs = {
        circle.name: disc_circles_dir / circle.name
        for circle in app_config.overlay.circles
        if (disc_circles_dir / circle.name).is_dir()
    }
    fovea_data = _load_fovea_overlay_data(output_path / "fovea.csv")

    batch_create_overlays(
        rgb_dir=rgb_dir,
        output_dir=overlay_path,
        av_dir=output_path / "artery_vein",
        disc_dir=disc_dir if disc_dir.is_dir() else None,
        vessels_dir=output_path / "vessels",
        circle_dirs=circle_dirs,
        vessel_width_data=df_vessel_widths,
        fovea_data=fovea_data,
        overlay_config=app_config.overlay,
    )

    df_selected_equivalent_widths = select_vessel_width_measurements_for_equivalents(
        df_vessel_widths,
        df_connection_widths,
    )
    equivalent_overlay_config = replace(
        app_config.overlay,
        layers=replace(app_config.overlay.layers, vessel_widths=True),
    )
    batch_create_overlays(
        rgb_dir=rgb_dir,
        output_dir=vessel_equivalent_overlay_path,
        av_dir=output_path / "artery_vein",
        disc_dir=disc_dir if disc_dir.is_dir() else None,
        vessels_dir=output_path / "vessels",
        circle_dirs=circle_dirs,
        vessel_width_data=df_selected_equivalent_widths,
        fovea_data=fovea_data,
        overlay_config=equivalent_overlay_config,
    )
    logger.info("Overlay regeneration complete in %s", output_path)


def _compute_and_save_vessel_metrics(
    vessels_path: Path,
    av_path: Path,
    disc_geometry_path: Path,
    output_path: Path,
    config_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        app_config = load_app_config(config_path)
        width_inner_circle, width_outer_circle = resolve_vessel_width_circle_pair(
            app_config.overlay.circles,
            inner_circle_name=app_config.vessel_widths.inner_circle,
            outer_circle_name=app_config.vessel_widths.outer_circle,
        )
        tortuosity_inner_circle, tortuosity_outer_circle = (
            resolve_vessel_width_circle_pair(
                app_config.overlay.circles,
                inner_circle_name=app_config.vessel_tortuosities.inner_circle,
                outer_circle_name=app_config.vessel_tortuosities.outer_circle,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    rgb_dir = None
    if app_config.vessel_widths.method == "profile":
        rgb_dir = _resolve_vessel_width_rgb_dir(
            output_path,
            app_config.vessel_widths.profile.image_source,
        )

    vessel_widths_path = output_path / "vessel_widths.csv"
    vessel_tortuosities_path = output_path / "vessel_tortuosities.csv"
    vessel_equivalents_path = output_path / "vessel_equivalents.csv"

    logger.info(
        "Measuring vessel widths between %s and %s with %d samples per connection using %s",
        width_inner_circle.name,
        width_outer_circle.name,
        app_config.vessel_widths.samples_per_connection,
        app_config.vessel_widths.method,
    )
    try:
        df_vessel_widths = measure_vessel_widths_between_disc_circle_pair(
            vessels_dir=vessels_path,
            av_dir=av_path,
            disc_geometry_path=disc_geometry_path,
            inner_circle=width_inner_circle,
            outer_circle=width_outer_circle,
            output_path=vessel_widths_path,
            samples_per_connection=app_config.vessel_widths.samples_per_connection,
            tangent_window_px=app_config.vessel_widths.profile.tangent_window_px,
            measurement_step_px=app_config.vessel_widths.profile.sample_step_px,
            width_config=app_config.vessel_widths,
            rgb_dir=rgb_dir,
        )
        logger.info(
            "Measuring vessel tortuosities between %s and %s",
            tortuosity_inner_circle.name,
            tortuosity_outer_circle.name,
        )
        df_vessel_tortuosities = measure_vessel_tortuosities_between_disc_circle_pair(
            vessels_dir=vessels_path,
            av_dir=av_path,
            disc_geometry_path=disc_geometry_path,
            inner_circle=tortuosity_inner_circle,
            outer_circle=tortuosity_outer_circle,
            output_path=vessel_tortuosities_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    df_connection_widths, df_vessel_equivalents = compute_revised_crx_from_widths(
        df_vessel_widths
    )
    df_vessel_equivalents.to_csv(vessel_equivalents_path, index=False)
    logger.info("Vessel equivalents saved to %s", vessel_equivalents_path)
    return (
        df_vessel_widths,
        df_vessel_tortuosities,
        df_connection_widths,
        df_vessel_equivalents,
    )


@click.group(name="vascx")
def cli():
    configure_logging()


@cli.command(name="vessel-metrics")
@click.argument(
    "source_output_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument("output_path", required=False, type=click.Path(path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a YAML config file. Defaults to ./config.yaml or the repo-root config.yaml when present.",
)
def vessel_metrics(
    source_output_path: Path,
    output_path: Path | None,
    config_path: Path | None,
):
    """Recompute vessel metrics from an existing pipeline output directory.

    SOURCE_OUTPUT_PATH must contain vessels/, artery_vein/, and disc_geometry.csv.
    Those intermediates are copied into OUTPUT_PATH before metrics are written. When
    OUTPUT_PATH is omitted, a timestamped output_YYYYMMDD_HHMMSS folder is created.
    """
    source_output_path = source_output_path.resolve()
    if output_path is None:
        output_path = _default_timestamped_output_path().resolve()
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

    _ensure_empty_or_new_output_dir(output_path)
    vessels_path, av_path, disc_geometry_path = (
        _copy_pipeline_output_for_vessel_metrics(
            source_output_path=source_output_path,
            output_path=output_path,
        )
    )
    _refresh_vessel_metric_disc_artifacts(
        output_path=output_path,
        config_path=config_path,
    )
    (
        df_vessel_widths,
        _,
        df_connection_widths,
        _,
    ) = _compute_and_save_vessel_metrics(
        vessels_path=vessels_path,
        av_path=av_path,
        disc_geometry_path=disc_geometry_path,
        output_path=output_path,
        config_path=config_path,
    )
    _refresh_vessel_metric_overlays(
        output_path=output_path,
        df_vessel_widths=df_vessel_widths,
        df_connection_widths=df_connection_widths,
        config_path=config_path,
    )
    logger.info("Vessel metrics complete. Results saved to %s", output_path)


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a YAML config file. Defaults to ./config.yaml or the repo-root config.yaml when present.",
)
@click.option(
    "--preprocess/--no-preprocess",
    default=True,
    help="Run preprocessing or use preprocessed images",
)
@click.option(
    "--vessels/--no-vessels", default=True, help="Run vessels and AV segmentation"
)
@click.option("--disc/--no-disc", default=True, help="Run optic disc segmentation")
@click.option(
    "--quality/--no-quality", default=True, help="Run image quality estimation"
)
@click.option("--fovea/--no-fovea", default=True, help="Run fovea detection")
@click.option(
    "--overlay/--no-overlay",
    default=None,
    help="Create visualization overlays. Defaults to the config value when set.",
)
@click.option(
    "--device",
    "device_name",
    type=click.Choice(["auto", "cuda", "mps", "cpu"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Inference device. 'auto' prefers CUDA first, then Apple Metal (MPS), then CPU.",
)
@click.option("--n_jobs", type=int, default=4, help="Number of preprocessing workers")
def run(
    data_path,
    output_path,
    config_path,
    preprocess,
    vessels,
    disc,
    quality,
    fovea,
    overlay,
    device_name,
    n_jobs,
):
    """Run the complete inference pipeline on fundus images.

    DATA_PATH is either a directory containing images or a CSV file with 'path' column.
    OUTPUT_PATH is the directory where results will be stored.
    """

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    try:
        app_config = load_app_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    overlay_enabled = app_config.overlay.enabled if overlay is None else overlay
    if app_config.source_path is not None:
        logger.info("Loaded config from %s", app_config.source_path)

    # Setup output directories
    preprocess_rgb_path = output_path / "preprocessed_rgb"
    vessels_path = output_path / "vessels"
    av_path = output_path / "artery_vein"
    disc_path = output_path / "disc"
    disc_circles_path = output_path / "disc_circles"
    overlay_path = output_path / "overlays"

    # Create required directories
    if preprocess:
        preprocess_rgb_path.mkdir(exist_ok=True, parents=True)
    if vessels:
        av_path.mkdir(exist_ok=True, parents=True)
        vessels_path.mkdir(exist_ok=True, parents=True)
    if disc:
        disc_path.mkdir(exist_ok=True, parents=True)
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
    vessel_equivalents_path = (
        output_path / "vessel_equivalents.csv" if disc and vessels else None
    )

    # Determine if input is a folder or CSV file
    data_path = Path(data_path)
    is_csv = data_path.suffix.lower() == ".csv"

    # Get files to process
    files = []
    ids = None

    if is_csv:
        logger.info("Reading file paths from CSV: %s", data_path)
        try:
            df = pd.read_csv(data_path)
            if "path" not in df.columns:
                logger.error("CSV must contain a 'path' column")
                return

            # Get file paths and convert to Path objects
            files = [Path(p) for p in df["path"]]

            if "id" in df.columns:
                ids = df["id"].tolist()
                logger.info("Using IDs from CSV 'id' column")

        except Exception as e:
            logger.exception("Error reading CSV file: %s", e)
            return
    else:
        logger.info("Finding files in directory: %s", data_path)
        files = list(data_path.glob("*"))
        ids = [f.stem for f in files]

    if not files:
        logger.warning("No files found to process")
        return

    logger.info("Found %d files to process", len(files))

    # Step 1: Preprocess images if requested
    if preprocess:
        logger.info("Running preprocessing")
        _run_preprocessing(
            files=files,
            ids=ids,
            rgb_path=preprocess_rgb_path,
            bounds_path=bounds_path,
            n_jobs=n_jobs,
        )
        # Use the preprocessed images for subsequent steps
        preprocessed_files = list(preprocess_rgb_path.glob("*.png"))
    else:
        # Use the input files directly
        preprocessed_files = files
    ids = [f.stem for f in preprocessed_files]
    logger.info("Prepared %d images for inference", len(preprocessed_files))

    available_devices = available_device_types()
    logger.info(
        "Device availability: cuda=%s, mps=%s, cpu=%s",
        available_devices["cuda"],
        available_devices["mps"],
        available_devices["cpu"],
    )
    try:
        device = resolve_device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if device_name == "auto":
        logger.info("Auto-selected device: %s", device)
    else:
        logger.info("Using requested device '%s': %s", device_name, device)

    # Step 2: Run quality estimation if requested
    if quality:
        logger.info("Running quality estimation")
        df_quality = run_quality_estimation(
            fpaths=preprocessed_files, ids=ids, device=device
        )
        df_quality.to_csv(quality_path)
        logger.info("Quality results saved to %s", quality_path)

    # Step 3: Run vessels and AV segmentation if requested
    if vessels:
        logger.info("Running vessels and AV segmentation")
        run_segmentation_vessels_and_av(
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

    # Step 4: Run optic disc segmentation if requested
    if disc:
        logger.info("Running optic disc segmentation")
        run_segmentation_disc(
            rgb_paths=preprocessed_files,
            ids=ids,
            output_path=disc_path,
            disc_color=app_config.overlay.colors.disc,
            device=device,
        )
        logger.info("Disc segmentation saved to %s", disc_path)
        generate_disc_circles(
            disc_dir=disc_path,
            circle_output_dir=disc_circles_path,
            circles=app_config.overlay.circles,
            measurements_path=disc_geometry_path,
        )
        logger.info("Disc circles saved to %s", disc_circles_path)

    # Step 5: Measure width and tortuosity outputs through separate vessel metric flows.
    df_vessel_widths = None
    df_selected_equivalent_widths = None
    if disc and vessels:
        (
            df_vessel_widths,
            df_vessel_tortuosities,
            df_connection_widths,
            _,
        ) = _compute_and_save_vessel_metrics(
            vessels_path=vessels_path,
            av_path=av_path,
            disc_geometry_path=disc_geometry_path,
            output_path=output_path,
            config_path=config_path,
        )
        df_selected_equivalent_widths = (
            select_vessel_width_measurements_for_equivalents(
                df_vessel_widths,
                df_connection_widths,
            )
        )

    # Step 6: Run fovea detection if requested
    df_fovea = None
    if fovea:
        logger.info("Running fovea detection")
        df_fovea = run_fovea_detection(
            rgb_paths=preprocessed_files, ids=ids, device=device
        )
        df_fovea.to_csv(fovea_path)
        logger.info("Fovea detection results saved to %s", fovea_path)

    # Step 7: Create overlays if requested
    if overlay_enabled:
        logger.info("Creating visualization overlays")

        # Prepare fovea data if available
        fovea_data = None
        if df_fovea is not None:
            fovea_data = {
                idx: (row["x_fovea"], row["y_fovea"])
                for idx, row in df_fovea.iterrows()
            }

        # Create visualization overlays
        batch_create_overlays(
            rgb_dir=preprocess_rgb_path if preprocess else data_path,
            output_dir=overlay_path,
            av_dir=av_path,
            disc_dir=disc_path,
            vessels_dir=vessels_path,
            circle_dirs={
                circle.name: disc_circles_path / circle.name
                for circle in app_config.overlay.circles
            },
            vessel_width_data=df_vessel_widths,
            fovea_data=fovea_data,
            overlay_config=app_config.overlay,
        )

        if df_selected_equivalent_widths is not None:
            vessel_equivalent_overlay_path = output_path / "vessel_equivalent_overlays"
            equivalent_overlay_config = replace(
                app_config.overlay,
                layers=replace(app_config.overlay.layers, vessel_widths=True),
            )
            batch_create_overlays(
                rgb_dir=preprocess_rgb_path if preprocess else data_path,
                output_dir=vessel_equivalent_overlay_path,
                av_dir=av_path,
                disc_dir=disc_path,
                vessels_dir=vessels_path,
                circle_dirs={
                    circle.name: disc_circles_path / circle.name
                    for circle in app_config.overlay.circles
                },
                vessel_width_data=df_selected_equivalent_widths,
                fovea_data=fovea_data,
                overlay_config=equivalent_overlay_config,
            )
            logger.info(
                "Vessel equivalent selection overlays saved to %s",
                vessel_equivalent_overlay_path,
            )

        logger.info("Visualization overlays saved to %s", overlay_path)

    logger.info("All requested processing complete. Results saved to %s", output_path)
