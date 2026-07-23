from pathlib import Path
from types import SimpleNamespace

import pytest

from identity_worker.errors import WorkerError
from identity_worker import runtime_preflight


def test_expected_dependency_lock_is_explicit():
    assert runtime_preflight.EXPECTED_VERSIONS == {
        "transformers": "4.56.2",
        "huggingface_hub": "0.35.1",
        "accelerate": "1.10.1",
        "tokenizers": "0.22.1",
        "peft": "0.17.1",
        "librosa": "0.11.0",
        "soundfile": "0.13.1",
        "soxr": "0.5.0.post1",
        "numba": "0.61.2",
        "scipy": "1.15.3",
    }


def test_runtime_preflight_accepts_complete_runtime(monkeypatch, tmp_path: Path):
    train_script = tmp_path / "examples" / "wanvideo" / "model_training" / "train.py"
    train_script.parent.mkdir(parents=True)
    train_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda name: runtime_preflight.EXPECTED_VERSIONS[name])
    monkeypatch.setattr(runtime_preflight, "_import_core_modules", lambda: {
        "accelerate_cli": SimpleNamespace(main=lambda: None),
        "transformers_hub": SimpleNamespace(is_offline_mode=lambda: False),
    })
    monkeypatch.setattr(runtime_preflight, "_load_training_entrypoint", lambda path: SimpleNamespace(wan_parser=lambda: None, WanTrainingModule=type("WanTrainingModule", (), {})))
    monkeypatch.setattr(runtime_preflight, "_probe_audio_operator", lambda: {"sourceRate": 8000, "targetRate": 16000, "sampleCount": 320})
    result = runtime_preflight.inspect_runtime(tmp_path)
    assert result["status"] == "IDENTITY_LORA_TRAINING_RUNTIME_READY"
    assert result["entrypoint"]["wanParser"] is True
    assert result["audioProbe"]["sampleCount"] == 320


def test_runtime_preflight_rejects_version_drift(monkeypatch):
    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda name: "0.0.0" if name == "huggingface_hub" else runtime_preflight.EXPECTED_VERSIONS[name])
    with pytest.raises(WorkerError) as captured:
        runtime_preflight.inspect_runtime()
    assert captured.value.code == "TRAINING_RUNTIME_VERSION_MISMATCH"
    assert captured.value.retryable is True


def test_missing_audio_distribution_has_specific_code(monkeypatch):
    def fake_version(name):
        if name == "librosa":
            raise runtime_preflight.metadata.PackageNotFoundError(name)
        return runtime_preflight.EXPECTED_VERSIONS[name]
    monkeypatch.setattr(runtime_preflight.metadata, "version", fake_version)
    with pytest.raises(WorkerError) as captured:
        runtime_preflight.inspect_runtime()
    assert captured.value.code == "TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING"


def test_runtime_preflight_classifies_core_import_failure(monkeypatch):
    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda name: runtime_preflight.EXPECTED_VERSIONS[name])
    monkeypatch.setattr(runtime_preflight, "_import_core_modules", lambda: (_ for _ in ()).throw(WorkerError("TRAINING_RUNTIME_IMPORT_FAILED", "broken", retryable=True)))
    with pytest.raises(WorkerError) as captured:
        runtime_preflight.inspect_runtime()
    assert captured.value.code == "TRAINING_RUNTIME_IMPORT_FAILED"


def test_entrypoint_failure_is_fail_closed(monkeypatch, tmp_path: Path):
    train_script = tmp_path / "examples" / "wanvideo" / "model_training" / "train.py"
    train_script.parent.mkdir(parents=True)
    train_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda name: runtime_preflight.EXPECTED_VERSIONS[name])
    monkeypatch.setattr(runtime_preflight, "_import_core_modules", lambda: {"accelerate_cli": SimpleNamespace(main=lambda: None), "transformers_hub": SimpleNamespace(is_offline_mode=lambda: False)})
    monkeypatch.setattr(runtime_preflight, "_load_training_entrypoint", lambda path: (_ for _ in ()).throw(WorkerError("TRAINING_RUNTIME_ENTRYPOINT_IMPORT_FAILED", "broken", retryable=True)))
    with pytest.raises(WorkerError) as captured:
        runtime_preflight.inspect_runtime(tmp_path)
    assert captured.value.code == "TRAINING_RUNTIME_ENTRYPOINT_IMPORT_FAILED"


def test_audio_probe_failure_is_fail_closed(monkeypatch, tmp_path: Path):
    train_script = tmp_path / "examples" / "wanvideo" / "model_training" / "train.py"
    train_script.parent.mkdir(parents=True)
    train_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda name: runtime_preflight.EXPECTED_VERSIONS[name])
    monkeypatch.setattr(runtime_preflight, "_import_core_modules", lambda: {"accelerate_cli": SimpleNamespace(main=lambda: None), "transformers_hub": SimpleNamespace(is_offline_mode=lambda: False)})
    monkeypatch.setattr(runtime_preflight, "_load_training_entrypoint", lambda path: SimpleNamespace(wan_parser=lambda: None, WanTrainingModule=type("WanTrainingModule", (), {})))
    monkeypatch.setattr(runtime_preflight, "_probe_audio_operator", lambda: (_ for _ in ()).throw(WorkerError("TRAINING_RUNTIME_AUDIO_PROBE_FAILED", "broken", retryable=True)))
    with pytest.raises(WorkerError) as captured:
        runtime_preflight.inspect_runtime(tmp_path)
    assert captured.value.code == "TRAINING_RUNTIME_AUDIO_PROBE_FAILED"
