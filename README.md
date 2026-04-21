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
```

Device selection defaults to `auto` (prefers `cuda`, then `mps`, then `cpu`). Requesting an unavailable accelerator exits with an error rather than silently falling back.

## Configuration

`config.yaml` controls overlay behaviour, disc-circle generation, and vessel-width sampling. The file is resolved from the current working directory first, then the repository root, or passed explicitly:

```bash
python -m vascx_models run DATA_PATH OUTPUT_PATH --config /path/to/config.yaml
```

The repository ships with this `config.yaml`:

```yaml
overlay:
  enabled: true
  layers:
    arteries: true
    veins: true
    disc: true
    fovea: true
    vessel_widths: true
  colours:
    artery: "#FF0000"
    vein: "#0000FF"
    vessel: "#00FF00"
    disc: "#FFFFFF"
    fovea: "#FFFF00"
    vessel_widths: "#00FF00"
  circles:
    - name: "2r"
      diameter: 2.0
      color: "#00FF00"
    - name: "3r"
      diameter: 3.0
      color: "#00FF00"
vessel_widths:
  inner_circle: "2r"
  outer_circle: "3r"
  samples_per_connection: 5
```

## Outputs

```text
OUTPUT_PATH/
├── preprocessed_rgb/
├── vessels/
├── artery_vein/
├── disc/
├── disc_circles/      # one subdirectory per configured circle name
├── overlays/
├── bounds.csv
├── disc_geometry.csv
├── vessel_widths.csv
├── quality.csv
└── fovea.csv
```

## Testing

```bash
pytest                                                                    # unit + CLI tests
pytest tests/test_e2e.py -q -k cpu                                      # real-model smoke test
```

The e2e test skips automatically when model weights have not been downloaded.
