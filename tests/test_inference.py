import pytest
import torch

from vascx_models.models import inference


def test_resolve_device_auto_prefers_cuda_then_mps_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inference.torch.backends.mps, "is_available", lambda: True)
    assert inference.resolve_device("auto") == torch.device("cuda:0")

    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference.torch.backends.mps, "is_available", lambda: True)
    assert inference.resolve_device("auto") == torch.device("mps")

    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference.torch.backends.mps, "is_available", lambda: False)
    assert inference.resolve_device("auto") == torch.device("cpu")


def test_resolve_device_rejects_unavailable_requested_accelerator(monkeypatch) -> None:
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="Requested device 'cuda' is not available"):
        inference.resolve_device("cuda")

    with pytest.raises(RuntimeError, match="Requested device 'mps' is not available"):
        inference.resolve_device("mps")


def test_run_quality_estimation_fails_fast_when_weights_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        inference,
        "ensure_model_files_present",
        lambda filenames: (_ for _ in ()).throw(FileNotFoundError("missing weights")),
    )

    with pytest.raises(FileNotFoundError, match="missing weights"):
        inference.run_quality_estimation([], ids=[], device=torch.device("cpu"))
