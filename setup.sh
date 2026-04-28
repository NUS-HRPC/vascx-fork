#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-vascx-fork}"
MODEL_REPO="${MODEL_REPO:-Eyened/vascx}"
MODEL_REVISION="${MODEL_REVISION:-latest}"
MODEL_SCOPE="${MODEL_SCOPE:-all}"
SKIP_ENV="${SKIP_ENV:-0}"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

Options:
  --env-name NAME          Conda environment name (default: vascx-fork)
  --model-repo REPO        Hugging Face model repo (default: Eyened/vascx)
  --model-revision REV     latest, tested, or a specific revision (default: latest)
  --latest                 Shortcut for --model-revision latest
  --tested                 Shortcut for --model-revision tested
  --core-only              Download only the runtime-required weights
  --skip-env               Skip conda environment creation/update
  -h, --help               Show this help message

Environment overrides:
  CONDA_ENV, MODEL_REPO, MODEL_REVISION, MODEL_SCOPE, SKIP_ENV, HF_TOKEN
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      CONDA_ENV="$2"
      shift 2
      ;;
    --model-repo)
      MODEL_REPO="$2"
      shift 2
      ;;
    --model-revision)
      MODEL_REVISION="$2"
      shift 2
      ;;
    --latest)
      MODEL_REVISION="latest"
      shift
      ;;
    --tested)
      MODEL_REVISION="tested"
      shift
      ;;
    --core-only)
      MODEL_SCOPE="core"
      shift
      ;;
    --skip-env)
      SKIP_ENV="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to run setup.sh" >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ "$SKIP_ENV" != "1" ]]; then
  if conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    echo "Updating conda environment '$CONDA_ENV'"
    conda env update --name "$CONDA_ENV" --file "$REPO_ROOT/environment.yml" --prune
  else
    echo "Creating conda environment '$CONDA_ENV'"
    conda env create --name "$CONDA_ENV" --file "$REPO_ROOT/environment.yml"
  fi
fi

download_args=(
  python
  -m
  vascx_models.models.model_assets
  download
  --repo-root
  "$REPO_ROOT"
  --model-repo
  "$MODEL_REPO"
  --model-revision
  "$MODEL_REVISION"
)

if [[ "$MODEL_SCOPE" == "core" ]]; then
  download_args+=(--core-only)
fi

echo "Downloading model weights from $MODEL_REPO (revision: $MODEL_REVISION, scope: $MODEL_SCOPE)"
conda run -n "$CONDA_ENV" --no-capture-output "${download_args[@]}"

echo "Verifying required runtime weights"
conda run -n "$CONDA_ENV" --no-capture-output \
  python -m vascx_models.models.model_assets verify --repo-root "$REPO_ROOT"
