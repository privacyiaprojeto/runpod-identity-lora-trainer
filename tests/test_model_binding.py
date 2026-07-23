from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from identity_worker.errors import WorkerError
from identity_worker import model_lock

REPOSITORY = "Wan-AI/Wan2.1-VACE-14B"
REVISION = "539c162b1387eac9dc4c20bd3f74671309e76a4c"


def _request_and_files(tmp_path: Path):
    snapshot = tmp_path / "models--Wan-AI--Wan2.1-VACE-14B" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    names = [
        *(f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors" for index in range(1, 8)),
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ]
    artifacts = []
    paths = {}
    for index, name in enumerate(names):
        path = snapshot / name
        payload = f"artifact-{index}-{name}".encode()
        path.write_bytes(payload)
        paths[name] = path
        artifacts.append({
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        })
    request = SimpleNamespace(payload={"model": {
        "repository": REPOSITORY,
        "revision": REVISION,
        "artifacts": list(reversed(artifacts)),
    }})
    settings = SimpleNamespace(model_cache_root=tmp_path, hf_token="")
    return request, settings, paths


def test_materialize_model_groups_all_seven_diffusion_shards(monkeypatch, tmp_path: Path):
    request, settings, paths = _request_and_files(tmp_path)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(paths[Path(kwargs["filename"]).name])

    monkeypatch.setattr(model_lock, "hf_hub_download", fake_download)
    binding = model_lock.materialize_model(request, settings)

    assert binding.repository == REPOSITORY
    assert binding.revision == REVISION
    assert len(binding.diffusion_shards) == 7
    assert [Path(item).name for item in binding.diffusion_shards] == [
        f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors"
        for index in range(1, 8)
    ]
    model_paths = binding.diffsynth_model_paths()
    assert isinstance(model_paths[0], list)
    assert len(model_paths[0]) == 7
    assert Path(model_paths[1]).name == "models_t5_umt5-xxl-enc-bf16.pth"
    assert Path(model_paths[2]).name == "Wan2.1_VAE.pth"
    assert all(call["repo_id"] == REPOSITORY for call in calls)
    assert all(call["revision"] == REVISION for call in calls)
    assert all(call["local_files_only"] is True for call in calls)


def test_materialize_model_rejects_repository_drift(monkeypatch, tmp_path: Path):
    request, settings, _ = _request_and_files(tmp_path)
    request.payload["model"]["repository"] = "Other/Model"
    monkeypatch.setattr(model_lock, "hf_hub_download", lambda **kwargs: "unused")
    with pytest.raises(WorkerError) as captured:
        model_lock.materialize_model(request, settings)
    assert captured.value.code == "MODEL_BINDING_REPOSITORY_MISMATCH"
    assert captured.value.retryable is False


def test_materialize_model_rejects_incomplete_shard_set(monkeypatch, tmp_path: Path):
    request, settings, paths = _request_and_files(tmp_path)
    request.payload["model"]["artifacts"] = [
        item for item in request.payload["model"]["artifacts"]
        if "00007-of-00007" not in item["path"]
    ]
    # Preserve nine items so validation reaches the strict shard check.
    duplicate = dict(request.payload["model"]["artifacts"][0])
    duplicate["path"] = "unexpected.bin"
    duplicate_path = next(iter(paths.values())).parent / "unexpected.bin"
    duplicate_path.write_bytes(b"unexpected")
    duplicate["sha256"] = hashlib.sha256(b"unexpected").hexdigest()
    request.payload["model"]["artifacts"].append(duplicate)

    def fake_download(**kwargs):
        filename = Path(kwargs["filename"]).name
        return str(duplicate_path if filename == "unexpected.bin" else paths[filename])

    monkeypatch.setattr(model_lock, "hf_hub_download", fake_download)
    with pytest.raises(WorkerError) as captured:
        model_lock.materialize_model(request, settings)
    assert captured.value.code == "MODEL_BINDING_SHARDS_INVALID"


def test_materialize_model_never_falls_back_to_network(monkeypatch, tmp_path: Path):
    request, settings, _ = _request_and_files(tmp_path)

    def missing(**kwargs):
        assert kwargs["local_files_only"] is True
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(model_lock, "hf_hub_download", missing)
    with pytest.raises(WorkerError) as captured:
        model_lock.materialize_model(request, settings)
    assert captured.value.code == "MODEL_CACHE_MISS"
    assert captured.value.retryable is True

def test_model_lock_module_does_not_import_huggingface_hub_eagerly():
    # A dependência é obrigatória no container, mas não deve ser exigida durante
    # a coleta dos testes locais que usam doubles para o cache privado.
    source = Path(model_lock.__file__).read_text(encoding="utf-8")
    assert "from huggingface_hub import hf_hub_download\n" not in source
    assert "def hf_hub_download(**kwargs):" in source

