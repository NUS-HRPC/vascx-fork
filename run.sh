#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-vascx-fork}"
SAMPLE_INPUT_PATH="$REPO_ROOT/samples/fundus/original"
DEFAULT_INPUT_PATH="images"
INPUT_PATH="${INPUT_PATH:-$DEFAULT_INPUT_PATH}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
DEFAULT_OUTPUT_PATH="$REPO_ROOT/output_$TIMESTAMP"
OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}"
N_JOBS="${N_JOBS:-1}"
DEVICE="${DEVICE:-auto}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample-run)
      INPUT_PATH="$SAMPLE_INPUT_PATH"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--sample-run]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$INPUT_PATH" ]]; then
  echo "Input directory does not exist: $INPUT_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_PATH"

echo "Running VascX Fork"
echo "  conda env:   $CONDA_ENV"
echo "  input path:  $INPUT_PATH"
echo "  output path: $OUTPUT_PATH"
echo "  n_jobs:      $N_JOBS"
echo "  device:      $DEVICE"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$REPO_ROOT"
exec python -m vascx_models run "$INPUT_PATH" "$OUTPUT_PATH" --n_jobs "$N_JOBS" --device "$DEVICE"
