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
    monkeypatch.setattr(runtime_preflight, "_probe_model_loader_contract", lambda: {"hashModelFile": True, "wanVideoVaceRegistered": True, "groupedPathProbeDeferredToRequest": True, "weightsLoaded": False})
    monkeypatch.setattr(runtime_preflight, "_probe_preview_runtime_contract", lambda: {"ffmpeg": True, "wanVideoPipeline": True, "vaceInputs": True, "loraLoader": True, "weightsLoaded": False, "gpuStarted": False})
    result = runtime_preflight.inspect_runtime(tmp_path)
    assert result["status"] == "IDENTITY_LORA_TRAINING_RUNTIME_READY"
    assert result["entrypoint"]["wanParser"] is True
    assert result["audioProbe"]["sampleCount"] == 320
    assert result["modelLoaderContract"]["wanVideoVaceRegistered"] is True
    assert result["previewRuntimeContract"]["vaceInputs"] is True
    assert result["previewRuntimeContract"]["gpuStarted"] is False


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


def test_model_loader_contract_requires_vace_registry(monkeypatch):
    def fake_import(name):
        if name == "diffsynth.core.loader":
            return SimpleNamespace(hash_model_file=lambda paths: "hash")
        if name == "diffsynth.configs":
            return SimpleNamespace(MODEL_CONFIGS=[{"model_hash": "x", "model_name": "other"}])
        raise AssertionError(name)
    monkeypatch.setattr(runtime_preflight.importlib, "import_module", fake_import)
    with pytest.raises(WorkerError) as captured:
        runtime_preflight._probe_model_loader_contract()
    assert captured.value.code == "TRAINING_MODEL_REGISTRY_MISSING_VACE"


def test_model_loader_contract_accepts_registered_vace(monkeypatch):
    def fake_import(name):
        if name == "diffsynth.core.loader":
            return SimpleNamespace(hash_model_file=lambda paths: "hash")
        if name == "diffsynth.configs":
            return SimpleNamespace(MODEL_CONFIGS=[{"model_hash": "hash", "model_name": "wan_video_vace"}])
        raise AssertionError(name)
    monkeypatch.setattr(runtime_preflight.importlib, "import_module", fake_import)
    result = runtime_preflight._probe_model_loader_contract()
    assert result["hashModelFile"] is True
    assert result["wanVideoVaceRegistered"] is True
    assert result["weightsLoaded"] is False


def test_preview_runtime_contract_accepts_required_api(monkeypatch):
    class FakePipeline:
        @staticmethod
        def from_pretrained():
            return None

        def load_lora(self):
            return None

        def __call__(self, vace_video=None, vace_reference_image=None, vace_scale=1.0, height=480, width=832, num_frames=17, num_inference_steps=20, seed=0, tiled=True):
            return []

    def fake_import(name):
        if name == "torch":
            return SimpleNamespace(bfloat16=object())
        if name == "PIL.Image":
            return SimpleNamespace(open=lambda path: None)
        if name == "diffsynth.pipelines.wan_video":
            return SimpleNamespace(WanVideoPipeline=FakePipeline, ModelConfig=lambda *args, **kwargs: None)
        if name == "diffsynth.utils.data":
            return SimpleNamespace(VideoData=lambda *args, **kwargs: [], save_video=lambda *args, **kwargs: None)
        raise AssertionError(name)

    monkeypatch.setattr(runtime_preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(runtime_preflight.importlib, "import_module", fake_import)
    result = runtime_preflight._probe_preview_runtime_contract()
    assert result["ffmpeg"] is True
    assert result["vaceInputs"] is True
    assert result["loraLoader"] is True
    assert result["weightsLoaded"] is False
    assert result["gpuStarted"] is False


def test_preview_runtime_contract_rejects_missing_vace_parameter(monkeypatch):
    class FakePipeline:
        @staticmethod
        def from_pretrained():
            return None

        def load_lora(self):
            return None

        def __call__(self, height=480, width=832):
            return []

    def fake_import(name):
        if name == "torch":
            return SimpleNamespace(bfloat16=object())
        if name == "PIL.Image":
            return SimpleNamespace(open=lambda path: None)
        if name == "diffsynth.pipelines.wan_video":
            return SimpleNamespace(WanVideoPipeline=FakePipeline, ModelConfig=lambda *args, **kwargs: None)
        if name == "diffsynth.utils.data":
            return SimpleNamespace(VideoData=lambda *args, **kwargs: [], save_video=lambda *args, **kwargs: None)
        raise AssertionError(name)

    monkeypatch.setattr(runtime_preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(runtime_preflight.importlib, "import_module", fake_import)
    with pytest.raises(WorkerError) as captured:
        runtime_preflight._probe_preview_runtime_contract()
    assert captured.value.code == "PREVIEW_RUNTIME_PIPELINE_INCOMPATIBLE"
