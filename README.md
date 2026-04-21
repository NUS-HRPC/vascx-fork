# VascX Fork

[![CI](https://github.com/zyf0717/vascx-fork/actions/workflows/ci.yml/badge.svg)](https://github.com/zyf0717/vascx-fork/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

`vascx-fork` is a GitHub-first fork of the original [`Eyened/vascx`](https://huggingface.co/Eyened/vascx) release.

This repository contains the Python package, runner scripts, tests, configuration, and sample inputs for the VascX retinal fundus analysis pipeline. Model weights are not committed here. They are downloaded on demand from Hugging Face with `./setup.sh`.

## Repository Layout

- Code and tests live in this GitHub repository.
- Model weights are downloaded into the expected local folders at setup time.
- The default model source is the upstream Hugging Face repo `Eyened/vascx`.
- The default model revision follows the upstream latest revision on `main`.
- A pinned tested revision is also available for reproducible setup.

### Model Weight Locations

`hf_hub_download` mirrors the Hugging Face repo's directory structure into the repository root, so the real `.pt` files land in categorised subdirectories:

```text
artery_vein/   av_july24.pt  av_july24_AVRDB.pt  ...
disc/          disc_july24.pt  disc_july24_ADAM.pt  ...
discedge/      discedge_july24.pt
fovea/         fovea_july24.pt
odfd/          odfd_march25.pt
quality/       quality.pt
vessels/       vessels_july24.pt  vessels_july24_DRHAGIS.pt  ...
```

At runtime, `configure_runtime_environment()` creates a flat `model_releases/` directory and populates it with symlinks pointing at those real files:

```text
model_releases/
├── av_july24.pt          -> ../artery_vein/av_july24.pt
├── disc_july24.pt        -> ../disc/disc_july24.pt
├── fovea_july24.pt       -> ../fovea/fovea_july24.pt
├── quality.pt            -> ../quality/quality.pt
├── vessels_july24.pt     -> ../vessels/vessels_july24.pt
└── ...
```

`model_releases/` is therefore a symlink view only — **the actual weight files are in the categorised subdirectories above**. Deleting `model_releases/` is safe; deleting the categorised subdirectories removes the weights and requires re-running `./setup.sh` to restore them. Both `model_releases/` and the `.pt` files in the categorised subdirectories are excluded from version control via `.gitignore`.

### HuggingFace Download Cache

`configure_runtime_environment()` redirects `XDG_CACHE_HOME` to a local `.cache/` directory inside the repository root, which keeps all HuggingFace activity out of `~/.cache/huggingface`. After `./setup.sh` runs, `.cache/huggingface/download/` mirrors the same categorised layout but contains only small `.metadata` files, one per downloaded asset:

```text
.cache/huggingface/download/
├── artery_vein/
│   ├── av_july24.pt.metadata       # revision hash, ETag, timestamp
│   ├── av_july24_AVRDB.pt.metadata
│   └── ...
├── disc/
├── vessels/
└── ...
```

Each metadata file holds three lines: the resolved git revision hash, a SHA-256 ETag, and a download timestamp. `hf_hub_download` uses these to skip re-downloading a file whose remote ETag has not changed. **No weight data is stored in `.cache/`** — only these metadata records. Deleting `.cache/` forces `./setup.sh` to re-check every file against the remote on the next run, but does not remove any weights.

## Quick Start

Create or update the environment and download the model weights:

```bash
./setup.sh
```

That command:

1. creates or updates the `vascx-fork` conda environment from `environment.yml`
2. installs the pipeline and downloader dependencies
3. downloads model weights from `Eyened/vascx`
4. verifies that the required runtime weights are present

Then run the sample pipeline:

```bash
./run.sh --sample-run
```

The standard Python entrypoint is:

```bash
python -m vascx_models run DATA_PATH OUTPUT_PATH
```

Both entrypoints auto-configure the local cache and model-release directories from the repository checkout.

## Setup Options

Download the latest upstream weights:

```bash
./setup.sh
```

Pin to the tested model revision from April 21, 2026:

```bash
./setup.sh --tested
```

Use a specific revision or alternate Hugging Face repo:

```bash
./setup.sh --model-revision ff4d0be5d283d73fbdaaff1de3ed97d5be1e646a
./setup.sh --model-repo zyf0717/vascx-fork --tested
```

Download only the weights required by the main runtime pipeline:

```bash
./setup.sh --core-only
```

Skip environment creation if you already have a suitable environment:

```bash
SKIP_ENV=1 ./setup.sh --tested
```

`setup.sh` also respects `CONDA_ENV`, `MODEL_REPO`, `MODEL_REVISION`, `MODEL_SCOPE`, `SKIP_ENV`, and `HF_TOKEN`.

## What Changed In This Fork

- Repository identity is `vascx-fork`
- The default conda environment name in `environment.yml`, `setup.sh`, and `run.sh` is `vascx-fork`
- The legacy `setup.py` and installed `vascx` console script were removed
- Supported entrypoints are `./run.sh` and `python -m vascx_models`
- Overlay generation can be configured from the root `config.yaml`
- Inference device selection is automatic by default and can be overridden explicitly
- Downloaded weights, generated outputs, caches, and other non-repository artifacts are excluded from version control

## Running The Pipeline

`run.sh` activates the `vascx-fork` conda environment, defaults to the bundled sample images, and writes to a timestamped `output_YYYYMMDD_HHMMSS/` directory.

Typical examples:

```bash
./run.sh --sample-run
INPUT_PATH=/path/to/images OUTPUT_PATH=/path/to/output N_JOBS=4 ./run.sh
DEVICE=cpu INPUT_PATH=/path/to/images OUTPUT_PATH=/path/to/output ./run.sh
python -m vascx_models run /path/to/images /path/to/output
python -m vascx_models run /path/to/image_list.csv /path/to/output
python -m vascx_models run /path/to/preprocessed/images /path/to/output --no-preprocess
python -m vascx_models run /path/to/images /path/to/output --device auto
python -m vascx_models run /path/to/images /path/to/output --device cpu
python -m vascx_models run /path/to/images /path/to/output --no-disc --no-quality --no-fovea --no-overlay
python -m vascx_models run /path/to/images /path/to/output --no-vessels
```

`DATA_PATH` can be:

- a directory of fundus images
- a CSV file with a `path` column

If the required weights are missing, the runtime fails with a clear message that points back to `./setup.sh`.

## Device Selection

Inference device selection is automatic by default.

- `--device auto` is the default for `python -m vascx_models run`
- `DEVICE=auto` is the default for `./run.sh`
- Auto-selection priority is `cuda` first, then Apple Metal `mps`, then `cpu`
- The CLI logs detected availability as `cuda=...`, `mps=...`, `cpu=True`
- The CLI also logs the selected device for each run
- You can force a backend with `--device cuda`, `--device mps`, or `--device cpu`
- `./run.sh` forwards the `DEVICE` environment variable to the Python CLI
- If you request `cuda` or `mps` explicitly and that backend is unavailable, the run exits with a clear error instead of silently falling back

## Configuration

This fork adds a root-level `config.yaml` for overlay behavior, disc-circle generation, and vessel-width sampling.

If `config.yaml` exists in the current working directory, it is loaded first. Otherwise the repository-root `config.yaml` is used when present. You can also pass a specific file:

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

With the default pipeline settings, `OUTPUT_PATH` contains:

```text
OUTPUT_PATH/
├── preprocessed_rgb/
├── vessels/
├── artery_vein/
├── disc/
├── disc_circles/
├── overlays/
├── bounds.csv
├── disc_geometry.csv
├── vessel_widths.csv
├── quality.csv
└── fovea.csv
```

`disc_circles/` contains one subdirectory per configured circle name.

## Repository Contents

- `vascx_models/`: package source and CLI
- `config.yaml`: fork-specific overlay configuration
- `environment.yml`: conda environment definition
- `setup.sh`: environment bootstrap and model downloader
- `run.sh`: primary local runner
- `tests/`: pytest suite
- `samples/`: bundled sample fundus images
- `notebooks/`: preprocessing and inference examples

## Testing

The test suite includes unit tests, CLI tests, and an opt-in real-model single-image end-to-end smoke test in `tests/test_e2e.py`.

Useful commands:

```bash
conda run -n vascx-fork pytest
KMP_DUPLICATE_LIB_OK=TRUE conda run -n vascx-fork pytest tests/test_e2e.py -q
KMP_DUPLICATE_LIB_OK=TRUE VASCX_RUN_E2E=1 conda run -n vascx-fork pytest tests/test_e2e.py -q -k cpu
```

`tests/test_e2e.py` now skips cleanly when the external model weights have not been downloaded yet.

## Upstream Reference

- Upstream Hugging Face repo: <https://huggingface.co/Eyened/vascx>
- Original paper: <https://arxiv.org/abs/2409.16016>
