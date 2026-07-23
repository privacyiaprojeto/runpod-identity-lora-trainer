from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from identity_worker.errors import WorkerError
from identity_worker.model_lock import MaterializedModelBinding
from identity_worker import model_preflight


def _binding(tmp_path: Path) -> MaterializedModelBinding:
    paths = []
    for index in range(1, 8):
        path = tmp_path / f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors"
        path.write_bytes(b"safetensors-header")
        paths.append(str(path))
    text_encoder = tmp_path / "models_t5_umt5-xxl-enc-bf16.pth"
    vae = tmp_path / "Wan2.1_VAE.pth"
    text_encoder.write_bytes(b"text")
    vae.write_bytes(b"vae")
    return MaterializedModelBinding(
        repository="Wan-AI/Wan2.1-VACE-14B",
        revision="539c162b1387eac9dc4c20bd3f74671309e76a4c",
        diffusion_shards=tuple(paths),
        text_encoder_path=str(text_encoder),
        vae_path=str(vae),
    )


def test_model_preflight_hashes_the_grouped_checkpoint(monkeypatch, tmp_path: Path):
    binding = _binding(tmp_path)
    seen = {}

    def fake_hash(paths):
        seen["paths"] = paths
        return "grouped-vace-hash"

    def fake_import(name):
        if name == "diffsynth.core.loader":
            return SimpleNamespace(hash_model_file=fake_hash)
        if name == "diffsynth.configs":
            return SimpleNamespace(MODEL_CONFIGS=[
                {"model_hash": "grouped-vace-hash", "model_name": "wan_video_vace"},
            ])
        raise AssertionError(name)

    monkeypatch.setattr(model_preflight.importlib, "import_module", fake_import)
    result = model_preflight.assert_model_binding_compatible(binding)
    assert seen["paths"] == list(binding.diffusion_shards)
    assert result["status"] == "IDENTITY_LORA_MODEL_BINDING_READY"
    assert result["modelName"] == "wan_video_vace"
    assert result["diffusionShardCount"] == 7
    assert result["groupedModelPaths"] is True
    assert result["weightsLoadedByPreflight"] is False


def test_model_preflight_supports_registry_objects(monkeypatch, tmp_path: Path):
    binding = _binding(tmp_path)

    def fake_import(name):
        if name == "diffsynth.core.loader":
            return SimpleNamespace(hash_model_file=lambda paths: "object-hash")
        if name == "diffsynth.configs":
            return SimpleNamespace(MODEL_CONFIGS=[
                SimpleNamespace(model_hash="object-hash", model_name="wan_video_vace"),
            ])
        raise AssertionError(name)

    monkeypatch.setattr(model_preflight.importlib, "import_module", fake_import)
    result = model_preflight.assert_model_binding_compatible(binding)
    assert result["modelHash"] == "object-hash"


def test_model_preflight_fails_before_training_when_registry_does_not_match(monkeypatch, tmp_path: Path):
    binding = _binding(tmp_path)

    def fake_import(name):
        if name == "diffsynth.core.loader":
            return SimpleNamespace(hash_model_file=lambda paths: "unknown-hash")
        if name == "diffsynth.configs":
            return SimpleNamespace(MODEL_CONFIGS=[
                {"model_hash": "known-hash", "model_name": "wan_video_vace"},
            ])
        raise AssertionError(name)

    monkeypatch.setattr(model_preflight.importlib, "import_module", fake_import)
    with pytest.raises(WorkerError) as captured:
        model_preflight.assert_model_binding_compatible(binding)
    assert captured.value.code == "TRAINING_MODEL_DETECTION_FAILED"
    assert captured.value.retryable is True


def test_model_preflight_wraps_loader_contract_errors(monkeypatch, tmp_path: Path):
    binding = _binding(tmp_path)
    monkeypatch.setattr(
        model_preflight.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("loader unavailable")),
    )
    with pytest.raises(WorkerError) as captured:
        model_preflight.assert_model_binding_compatible(binding)
    assert captured.value.code == "TRAINING_MODEL_PREFLIGHT_FAILED"
    assert captured.value.retryable is True
