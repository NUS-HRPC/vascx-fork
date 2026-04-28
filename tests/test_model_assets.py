from pathlib import Path

import pytest

from vascx_models.models import model_assets


def test_resolve_model_revision_supports_latest_and_tested_aliases() -> None:
    assert model_assets.resolve_model_revision(None) == model_assets.LATEST_MODEL_REVISION
    assert model_assets.resolve_model_revision("latest") == model_assets.LATEST_MODEL_REVISION
    assert (
        model_assets.resolve_model_revision("tested")
        == model_assets.TESTED_MODEL_REVISION
    )
    assert model_assets.resolve_model_revision("custom-rev") == "custom-rev"


def test_missing_model_paths_checks_required_assets_by_default(tmp_path: Path) -> None:
    missing_paths = model_assets.missing_model_paths(tmp_path)

    assert len(missing_paths) == len(model_assets.REQUIRED_MODEL_ASSETS)
    assert tmp_path / "odfd" / "odfd_march25.pt" not in missing_paths


def test_ensure_model_files_present_reports_missing_paths(tmp_path: Path) -> None:
    model_path = tmp_path / "quality" / "quality.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("ok", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="disc/disc_july24.pt"):
        model_assets.ensure_model_files_present(
            ["quality.pt", "disc_july24.pt"],
            root=tmp_path,
        )


def test_ensure_model_files_present_rejects_unknown_filenames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model filename"):
        model_assets.ensure_model_files_present(["not-a-real-model.pt"], root=tmp_path)
