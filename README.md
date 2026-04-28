# VascX Fork

[![CI](https://github.com/zyf0717/vascx-fork/actions/workflows/ci.yaml/badge.svg)](https://github.com/zyf0717/vascx-fork/actions/workflows/ci.yaml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

A GitHub-first fork of [`Eyened/vascx`](https://huggingface.co/Eyened/vascx) — a retinal fundus analysis pipeline. Model weights are not committed here; they are downloaded on demand via `./setup.sh`.

## Repository Layout

```text
vascx_models/      # package source and CLI
tests/             # pytest suite
samples/           # bundled sample fundus images
config.yaml        # overlay and pipeline configuration
environment.yml    # conda environment definition
setup.sh           # environment bootstrap and model downloader
run.sh             # primary local runner
```

Model weights are downloaded into categorised subdirectories:

```text
artery_vein/   av_july24.pt  av_july24_AVRDB.pt  ...
disc/          disc_july24.pt  disc_july24_ADAM.pt  ...
discedge/      discedge_july24.pt
fovea/         fovea_july24.pt
odfd/          odfd_march25.pt
quality/       quality.pt
vessels/       vessels_july24.pt  vessels_july24_DRHAGIS.pt  ...
```

At runtime, `configure_runtime_environment()` creates a flat `model_releases/` directory of symlinks pointing at the files above. Deleting `model_releases/` is safe; deleting the categorised subdirectories requires re-running `./setup.sh` to restore the weights. Both are excluded from version control.

## Quick Start

```bash
./setup.sh          # create conda env and download weights
./run.sh --sample-run
```

Or with the Python entrypoint:

```bash
python -m vascx_models run DATA_PATH OUTPUT_PATH
```

`DATA_PATH` can be a directory of fundus images or a CSV with a `path` column.

## Setup Options

```bash
./setup.sh                          # latest upstream weights
./setup.sh --tested                 # pinned revision (April 21, 2026)
./setup.sh --core-only              # runtime weights only
./setup.sh --model-revision <hash>
./setup.sh --model-repo zyf0717/vascx-fork --tested
SKIP_ENV=1 ./setup.sh --tested      # skip conda env creation
```

`setup.sh` also respects `CONDA_ENV`, `MODEL_REPO`, `MODEL_REVISION`, `MODEL_SCOPE`, `SKIP_ENV`, and `HF_TOKEN`.

## Running The Pipeline

Use any one of the following commands:

```bash
./run.sh --sample-run                                                                    # run on bundled sample images
INPUT_PATH=/path/to/images OUTPUT_PATH=/path/to/output N_JOBS=4 ./run.sh               # custom paths and parallelism
python -m vascx_models run /path/to/images /path/to/output                             # image directory input
python -m vascx_models run /path/to/image_list.csv /path/to/output                     # CSV with 'path' column
python -m vascx_models run /path/to/images /path/to/output --no-preprocess             # skip preprocessing step
python -m vascx_models run /path/to/images /path/to/output --device cpu                # force CPU inference
python -m vascx_models run /path/to/images /path/to/output --no-disc --no-quality --no-fovea --no-overlay  # vessels only
python -m vascx_models vessel-metrics /path/to/existing_output                            # writes output_YYYYMMDD_HHMMSS
python -m vascx_models vessel-metrics /path/to/existing_output /path/to/new_metrics_output # explicit metrics output path
```

Device selection defaults to `auto` (prefers `cuda`, then `mps`, then `cpu`). Requesting an unavailable accelerator exits with an error rather than silently falling back.

## Configuration

`config.yaml` controls overlay behaviour, disc-circle generation, and vessel-width sampling. The file is resolved from the current working directory first, then the repository root, or passed explicitly:

```bash
python -m vascx_models run DATA_PATH OUTPUT_PATH --config /path/to/config.yaml
```

The repository root ships with a fully commented [config.yaml](/Users/yifei/repos/vascx-fork/config.yaml). Use that file as the source of truth for the current defaults and the valid values for each setting.

Vessel widths are sampled between the configured inner and outer disc circles.
Each retained vessel connection receives `samples_per_connection` interior
measurements. `vessel_widths.method` selects the backend:

When `samples_per_connection` is positive, that many evenly spaced interior
measurements are taken per retained connection. When it is negative, the
pipeline measures every interior path pixel along the retained vessel trace,
excluding the start and end pixels.

- `mask`: existing subpixel mask-boundary method, unchanged and still the default.
- `pvbm_mask`: integer/grid mask-width baseline using the local path direction and a perpendicular normal.
- `profile`: green-channel image-profile-derived width estimation. By default this reads from `preprocessed_rgb/`; if those RGB images are unavailable the run fails clearly unless `vessel_widths.profile.fallback_to_mask` is enabled.

The `profile` backend uses the mask width only as a guardrail/reference. CRAE/CRVE aggregation continues to use the per-sample `width_px` values emitted by the active backend.

Width measurement remains trunk-focused:

- vessel components must be traceable from the inner circle to the outer circle
- dead-end side branches are discarded
- for a fork inside the annulus, only the pre-fork trunk segment is measured
- downstream daughter branches are not measured for width or CRAE/CRVE selection

Tortuosity measurement is handled separately and uses a different correspondence rule:

- artery and vein masks are traced independently inside the configured tortuosity annulus
- direct 1-to-1 segments are measured directly
- valid 1-to-many bifurcations are split into a trunk segment plus child segments
- ambiguous many-to-1 or many-to-many patterns are discarded
- tortuosity summaries are length-weighted by `path_length_px`

## Outputs

```text
OUTPUT_PATH/
├── preprocessed_rgb/
├── vessels/
├── artery_vein/
├── disc/
├── disc_circles/      # one subdirectory per configured circle name
├── overlays/
├── vessel_width_overlays/
├── vessel_tortuosity_overlays/
├── bounds.csv
├── disc_geometry.csv
├── vessel_widths.csv
├── vessel_tortuosities.csv
├── vessel_tortuosity_summary.csv
├── vessel_equivalents.csv
├── quality.csv
└── fovea.csv
```

Key CSV outputs:

- `quality.csv`: image quality scores.
- `fovea.csv`: detected fovea coordinates.
- `disc_geometry.csv`: optic disc center and radius used for circle generation.
- `vessel_widths.csv`: per-sample vessel width measurements, including measurement
  endpoints (`x_start`, `y_start`, `x_end`, `y_end`) for overlay rendering, plus
  method/provenance fields such as `width_method`, `normal_x`, `normal_y`,
  `mask_width_px`, `measurement_valid`, and `measurement_failure_reason`.
- `vessel_tortuosities.csv`: per-segment tortuosity records, computed as skeleton
  path length divided by endpoint chord length for each retained tortuosity segment.
- `vessel_tortuosity_summary.csv`: per-image, per-vessel-type tortuosity summary,
  including the number of retained segments, total retained path length, and a
  length-weighted mean tortuosity (`TORTA` for arteries, `TORTV` for veins).
- `vessel_equivalents.csv`: CRAE/CRVE summary per image and vessel type.

Additional `vessel_widths.csv` columns include:

- `profile_channel`, `profile_left_t`, `profile_right_t`, `profile_trough_t`
- `profile_trough_value`, `profile_background_value`, `profile_contrast`
- `profile_threshold`, `profile_confidence`

`vessel_equivalents.csv` includes:

- `metric`: `CRAE` for arteries or `CRVE` for veins.
- `requested_n_largest`: number of vessels requested for the equivalent formula,
  currently 6.
- `n_vessels_available`: retained measured vessels available for that image/type.
- `n_vessels_used`: number actually used. If fewer than 6 are available, all
  available vessels are used.
- `vessel_ids_used`: semicolon-separated IDs such as `artery_7;artery_3`.
- `mean_widths_used_px`: semicolon-separated mean vessel widths in pixels.
- `equivalent_px`: recursive Knudtson-Parr-Hubbard equivalent in pixels. This is
  `NaN` when fewer than two vessels are available for a type.

Overlay directories:

- `overlays/`: standard segmentation and measurement overlay.
- `vessel_width_overlays/`: highlights only the width measurements from
  vessels selected for CRAE/CRVE calculation.
- `vessel_tortuosity_overlays/`: processed RGB overlaid with a green vessel
  skeleton and the Euclidean endpoint chords for retained tortuosity segments.

The CRAE/CRVE values are pixel-space equivalents unless you apply an external
pixel-to-length calibration.

## Recomputing Vessel Metrics

`vessel-metrics` runs only the path, width, tortuosity, and CRAE/CRVE stages
from an existing pipeline output. The source directory must contain:

- `vessels/`
- `artery_vein/`
- `disc_geometry.csv`

The full source output directory is copied into the requested new output
directory, then `vessel_widths.csv`, `vessel_tortuosities.csv`,
`vessel_tortuosity_summary.csv`, `vessel_equivalents.csv`, and the overlay
directories are refreshed there. When no destination is provided,
`vessel-metrics` creates a standard `output_YYYYMMDD_HHMMSS` folder in the
current working directory. The destination must be new or empty, and it cannot
be inside the source output directory.

## Testing

```bash
pytest                                                                    # unit + CLI tests
pytest tests/test_e2e.py -q -k cpu                                      # real-model smoke test
```

The e2e test skips automatically when model weights have not been downloaded.
