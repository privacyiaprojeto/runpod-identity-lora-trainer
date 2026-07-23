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
    }


def test_runtime_preflight_accepts_compatible_modules(monkeypatch, tmp_path: Path):
    train_script = tmp_path / "examples" / "wanvideo" / "model_training" / "train.py"
    train_script.parent.mkdir(parents=True)
    train_script.write_text("# test", encoding="utf-8")

    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda name: runtime_preflight.EXPECTED_VERSIONS[name])

    def fake_import(name):
        if name == "accelerate.commands.accelerate_cli":
            return SimpleNamespace(main=lambda: None)
        if name == "transformers.utils.hub":
            return SimpleNamespace(is_offline_mode=lambda: False)
        return SimpleNamespace()

    monkeypatch.setattr(runtime_preflight.importlib, "import_module", fake_import)
    result = runtime_preflight.inspect_runtime(tmp_path)
    assert result["status"] == "IDENTITY_LORA_TRAINING_RUNTIME_READY"


def test_runtime_preflight_rejects_version_drift(monkeypatch):
    monkeypatch.setattr(
        runtime_preflight.metadata,
        "version",
        lambda name: "0.0.0" if name == "huggingface_hub" else runtime_preflight.EXPECTED_VERSIONS[name],
    )
    with pytest.raises(WorkerError) as captured:
        runtime_preflight.inspect_runtime()
    assert captured.value.code == "TRAINING_RUNTIME_VERSION_MISMATCH"
    assert captured.value.retryable is True


def test_runtime_preflight_classifies_import_failure(monkeypatch):
    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda name: runtime_preflight.EXPECTED_VERSIONS[name])
    monkeypatch.setattr(runtime_preflight.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("broken")))
    with pytest.raises(WorkerError) as captured:
        runtime_preflight.inspect_runtime()
    assert captured.value.code == "TRAINING_RUNTIME_IMPORT_FAILED"
    assert captured.value.retryable is True
