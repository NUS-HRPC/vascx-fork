from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_MODEL_REPO = "Eyened/vascx"
LATEST_MODEL_REVISION = "main"
TESTED_MODEL_REVISION = "ff4d0be5d283d73fbdaaff1de3ed97d5be1e646a"


@dataclass(frozen=True)
class ModelAsset:
    relative_path: str
    required: bool = False

    @property
    def path(self) -> Path:
        return Path(self.relative_path)

    @property
    def filename(self) -> str:
        return self.path.name


REQUIRED_MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset("artery_vein/av_july24.pt", required=True),
    ModelAsset("disc/disc_july24.pt", required=True),
    ModelAsset("fovea/fovea_july24.pt", required=True),
    ModelAsset("quality/quality.pt", required=True),
    ModelAsset("vessels/vessels_july24.pt", required=True),
)

OPTIONAL_MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset("artery_vein/av_july24_AVRDB.pt"),
    ModelAsset("artery_vein/av_july24_IOSTAR.pt"),
    ModelAsset("artery_vein/av_july24_LEUVEN.pt"),
    ModelAsset("artery_vein/av_july24_RS.pt"),
    ModelAsset("disc/disc_july24_ADAM.pt"),
    ModelAsset("disc/disc_july24_IDRID.pt"),
    ModelAsset("disc/disc_july24_ORIGA.pt"),
    ModelAsset("disc/disc_july24_PAPILA.pt"),
    ModelAsset("discedge/discedge_july24.pt"),
    ModelAsset("odfd/odfd_march25.pt"),
    ModelAsset("vessels/vessels_july24_DRHAGIS.pt"),
    ModelAsset("vessels/vessels_july24_FIVES.pt"),
    ModelAsset("vessels/vessels_july24_LEUVEN.pt"),
    ModelAsset("vessels/vessels_july24_RS.pt"),
)

ALL_MODEL_ASSETS: tuple[ModelAsset, ...] = REQUIRED_MODEL_ASSETS + OPTIONAL_MODEL_ASSETS
MODEL_ASSET_BY_FILENAME = {asset.filename: asset for asset in ALL_MODEL_ASSETS}


def all_model_assets(*, include_optional: bool = True) -> tuple[ModelAsset, ...]:
    if include_optional:
        return ALL_MODEL_ASSETS
    return REQUIRED_MODEL_ASSETS


def resolve_model_revision(model_revision: str | None) -> str:
    if model_revision in {None, "", "latest"}:
        return LATEST_MODEL_REVISION
    if model_revision == "tested":
        return TESTED_MODEL_REVISION
    return model_revision


def missing_model_paths(
    root: Path, *, include_optional: bool = False
) -> tuple[Path, ...]:
    return tuple(
        root / asset.path
        for asset in all_model_assets(include_optional=include_optional)
        if not (root / asset.path).exists()
    )


def ensure_model_files_present(
    filenames: Sequence[str], *, root: Path | None = None
) -> None:
    if root is None:
        from ..runtime import repo_root

        root = repo_root()

    unknown_filenames = [
        filename for filename in filenames if filename not in MODEL_ASSET_BY_FILENAME
    ]
    if unknown_filenames:
        joined = ", ".join(sorted(unknown_filenames))
        raise ValueError(f"Unknown model filename(s): {joined}")

    missing_paths = tuple(
        root / MODEL_ASSET_BY_FILENAME[filename].path
        for filename in filenames
        if not (root / MODEL_ASSET_BY_FILENAME[filename].path).exists()
    )
    if not missing_paths:
        return

    missing_display = ", ".join(str(path.relative_to(root)) for path in missing_paths)
    raise FileNotFoundError(
        "Missing required model weights: "
        f"{missing_display}. Run ./setup.sh or "
        "`python -m vascx_models.models.model_assets download --repo-root .`."
    )


def ensure_required_model_files(*, root: Path | None = None) -> None:
    ensure_model_files_present(
        [asset.filename for asset in REQUIRED_MODEL_ASSETS],
        root=root,
    )


def download_model_assets(
    *,
    root: Path,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str | None = "latest",
    include_optional: bool = True,
    token: str | None = None,
) -> tuple[Path, ...]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download model assets. "
            "Install dependencies from environment.yml first."
        ) from exc

    resolved_revision = resolve_model_revision(model_revision)
    downloaded_paths = []
    for asset in all_model_assets(include_optional=include_optional):
        target_dir = root / asset.path.parent
        target_dir.mkdir(exist_ok=True, parents=True)
        local_path = hf_hub_download(
            repo_id=model_repo,
            filename=asset.relative_path,
            revision=resolved_revision,
            local_dir=root,
            token=token,
        )
        downloaded_paths.append(Path(local_path))
    return tuple(downloaded_paths)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or verify external VascX model assets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download", help="Download model assets from Hugging Face."
    )
    download_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Local repository root where model directories should be populated.",
    )
    download_parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Hugging Face model repository to download from.",
    )
    download_parser.add_argument(
        "--model-revision",
        default="latest",
        help="Revision to download. Use 'latest', 'tested', or a specific commit hash/tag.",
    )
    download_parser.add_argument(
        "--core-only",
        action="store_true",
        help="Download only the weights required by the main inference pipeline.",
    )
    download_parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. HF_TOKEN is also respected automatically.",
    )

    verify_parser = subparsers.add_parser(
        "verify", help="Verify that the required runtime weights are present."
    )
    verify_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Local repository root containing downloaded model directories.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()

    if args.command == "download":
        downloaded_paths = download_model_assets(
            root=root,
            model_repo=args.model_repo,
            model_revision=args.model_revision,
            include_optional=not args.core_only,
            token=args.token,
        )
        resolved_revision = resolve_model_revision(args.model_revision)
        print(
            f"Downloaded {len(downloaded_paths)} model files from "
            f"{args.model_repo}@{resolved_revision} into {root}"
        )
        return 0

    missing_paths = missing_model_paths(root)
    if missing_paths:
        for path in missing_paths:
            print(f"MISSING {path.relative_to(root)}")
        return 1

    print("Required model weights are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
