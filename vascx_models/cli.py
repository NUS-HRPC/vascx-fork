import logging
import warnings
from datetime import datetime
from pathlib import Path

import click
from rtnls_fundusprep.cli import _run_preprocessing

from . import pipeline as pipeline_ops
from .config import load_app_config
from .geometry.disc_circles import generate_disc_circles
from .metrics.vessel_tortuosities import (
    measure_vessel_tortuosities_between_disc_circle_pair,
    summarize_vessel_tortuosities,
)
from .metrics.vessel_widths import (
    compute_revised_crx_from_widths,
    measure_vessel_widths_between_disc_circle_pair,
    resolve_vessel_width_circle_pair,
    select_vessel_width_measurements_for_equivalents,
)
from .models.inference import (
    available_device_types,
    resolve_device,
    run_fovea_detection,
    run_quality_estimation,
    run_segmentation_disc,
    run_segmentation_vessels_and_av,
)
from .overlays.utils import batch_create_overlays
from .runtime import configure_runtime_environment

configure_runtime_environment()

logger = logging.getLogger(__name__)


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


def _default_timestamped_output_path(base_dir: Path | None = None) -> Path:
    parent = Path.cwd() if base_dir is None else base_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return parent / f"output_{timestamp}"


def _pipeline_dependencies() -> pipeline_ops.PipelineDependencies:
    return pipeline_ops.PipelineDependencies(
        available_device_types=available_device_types,
        resolve_device=resolve_device,
        run_quality_estimation=run_quality_estimation,
        run_fovea_detection=run_fovea_detection,
        run_segmentation_vessels_and_av=run_segmentation_vessels_and_av,
        run_segmentation_disc=run_segmentation_disc,
        generate_disc_circles=generate_disc_circles,
        resolve_vessel_width_circle_pair=resolve_vessel_width_circle_pair,
        measure_vessel_widths_between_disc_circle_pair=measure_vessel_widths_between_disc_circle_pair,
        compute_revised_crx_from_widths=compute_revised_crx_from_widths,
        measure_vessel_tortuosities_between_disc_circle_pair=measure_vessel_tortuosities_between_disc_circle_pair,
        summarize_vessel_tortuosities=summarize_vessel_tortuosities,
        select_vessel_width_measurements_for_equivalents=select_vessel_width_measurements_for_equivalents,
        batch_create_overlays=batch_create_overlays,
        run_preprocessing=_run_preprocessing,
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
    try:
        app_config = load_app_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    pipeline_ops.run_vessel_metrics_pipeline(
        source_output_path=source_output_path,
        output_path=output_path,
        app_config=app_config,
        deps=_pipeline_dependencies(),
        default_timestamped_output_path=_default_timestamped_output_path,
    )


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
    try:
        app_config = load_app_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    pipeline_ops.run_pipeline(
        data_path=data_path,
        output_path=output_path,
        app_config=app_config,
        deps=_pipeline_dependencies(),
        preprocess=preprocess,
        vessels=vessels,
        disc=disc,
        quality=quality,
        fovea=fovea,
        overlay=overlay,
        device_name=device_name,
        n_jobs=n_jobs,
    )
