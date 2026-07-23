from __future__ import annotations

import argparse
import importlib
import json
import tempfile
import wave
from array import array
from importlib import metadata, util
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import WorkerError

EXPECTED_VERSIONS = {
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

AUDIO_DISTRIBUTIONS = frozenset({"librosa", "soundfile", "soxr", "numba", "scipy"})
AUDIO_IMPORTS = ("librosa", "soundfile", "soxr", "numba", "scipy")


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        code = "TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING" if distribution in AUDIO_DISTRIBUTIONS else "TRAINING_RUNTIME_DEPENDENCY_MISSING"
        raise WorkerError(
            code,
            f"Dependência obrigatória ausente: {distribution}.",
            retryable=True,
        ) from exc


def _import_core_modules() -> dict[str, ModuleType]:
    try:
        modules = {name: importlib.import_module(name) for name in AUDIO_IMPORTS}
        modules["transformers"] = importlib.import_module("transformers")
        modules["accelerate_cli"] = importlib.import_module("accelerate.commands.accelerate_cli")
        modules["transformers_hub"] = importlib.import_module("transformers.utils.hub")
        return modules
    except (ImportError, ModuleNotFoundError) as exc:
        missing = str(getattr(exc, "name", "") or "").split(".", 1)[0]
        code = "TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING" if missing in AUDIO_IMPORTS else "TRAINING_RUNTIME_IMPORT_FAILED"
        message = (
            "O runtime de áudio interno do trainer está incompleto."
            if code == "TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING"
            else "O runtime do treinamento possui dependências Python incompatíveis."
        )
        raise WorkerError(code, message, retryable=True) from exc


def _load_training_entrypoint(training_script: Path) -> ModuleType:
    try:
        spec = util.spec_from_file_location("privacy_identity_lora_training_entrypoint_preflight", training_script)
        if spec is None or spec.loader is None:
            raise ImportError("Não foi possível criar o loader do entrypoint oficial.")
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError) as exc:
        missing = str(getattr(exc, "name", "") or "").split(".", 1)[0]
        code = "TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING" if missing in AUDIO_IMPORTS else "TRAINING_RUNTIME_ENTRYPOINT_IMPORT_FAILED"
        raise WorkerError(
            code,
            "O entrypoint oficial do DiffSynth não conseguiu carregar todas as dependências do treinamento.",
            retryable=True,
        ) from exc
    except Exception as exc:
        raise WorkerError(
            "TRAINING_RUNTIME_ENTRYPOINT_IMPORT_FAILED",
            "O entrypoint oficial do DiffSynth falhou durante o preflight.",
            retryable=True,
        ) from exc

    if not callable(getattr(module, "wan_parser", None)) or not isinstance(getattr(module, "WanTrainingModule", None), type):
        raise WorkerError(
            "TRAINING_RUNTIME_ENTRYPOINT_INVALID",
            "O entrypoint oficial do treinamento não corresponde ao contrato homologado.",
            retryable=True,
        )
    return module


def _probe_audio_operator() -> dict[str, int]:
    try:
        operators = importlib.import_module("diffsynth.core.data.operators")
        load_audio = getattr(operators, "LoadAudio")
        loader = load_audio(sr=16000)
        with tempfile.TemporaryDirectory(prefix="identity_lora_audio_probe_") as temp:
            probe_path = Path(temp) / "probe.wav"
            # Fonte em 8 kHz força o caminho real de leitura + reamostragem para 16 kHz.
            with wave.open(str(probe_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(array("h", [0] * 160).tobytes())
            samples = loader(str(probe_path))
        sample_count = int(len(samples))
        if sample_count <= 0:
            raise RuntimeError("O probe de áudio retornou zero amostras.")
        return {"sourceRate": 8000, "targetRate": 16000, "sampleCount": sample_count}
    except (ImportError, ModuleNotFoundError) as exc:
        raise WorkerError(
            "TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING",
            "O runtime interno de áudio exigido pelo DiffSynth está incompleto.",
            retryable=True,
        ) from exc
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError(
            "TRAINING_RUNTIME_AUDIO_PROBE_FAILED",
            "O runtime de áudio não conseguiu executar a leitura e reamostragem de segurança.",
            retryable=True,
        ) from exc


def _probe_model_loader_contract() -> dict[str, Any]:
    try:
        loader = importlib.import_module("diffsynth.core.loader")
        configs = importlib.import_module("diffsynth.configs")
        hash_model_file = getattr(loader, "hash_model_file")
        model_configs = list(getattr(configs, "MODEL_CONFIGS"))
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError) as exc:
        raise WorkerError(
            "TRAINING_MODEL_LOADER_CONTRACT_INVALID",
            "O runtime do DiffSynth não expõe o contrato homologado de detecção de modelos.",
            retryable=True,
        ) from exc

    if not callable(hash_model_file):
        raise WorkerError(
            "TRAINING_MODEL_LOADER_CONTRACT_INVALID",
            "O detector de modelos do DiffSynth não está disponível.",
            retryable=True,
        )

    def config_value(item: Any, key: str) -> Any:
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    registered_names = {
        str(config_value(item, "model_name") or "").strip()
        for item in model_configs
    }
    if "wan_video_vace" not in registered_names:
        raise WorkerError(
            "TRAINING_MODEL_REGISTRY_MISSING_VACE",
            "O registry do DiffSynth não contém o modelo Wan VACE homologado.",
            retryable=True,
        )

    return {
        "hashModelFile": True,
        "wanVideoVaceRegistered": True,
        "groupedPathProbeDeferredToRequest": True,
        "weightsLoaded": False,
    }

def inspect_runtime(diffsynth_root: Path | None = None) -> dict[str, Any]:
    versions = {name: _version(name) for name in EXPECTED_VERSIONS}
    mismatches = {
        name: {"expected": expected, "actual": versions[name]}
        for name, expected in EXPECTED_VERSIONS.items()
        if versions[name] != expected
    }
    if mismatches:
        raise WorkerError(
            "TRAINING_RUNTIME_VERSION_MISMATCH",
            "O runtime do treinamento não corresponde ao conjunto de dependências homologado.",
            retryable=True,
        )

    modules = _import_core_modules()
    accelerate_cli = modules["accelerate_cli"]
    transformers_hub = modules["transformers_hub"]
    if not callable(getattr(accelerate_cli, "main", None)):
        raise WorkerError(
            "TRAINING_RUNTIME_ACCELERATE_INVALID",
            "O comando accelerate não está disponível no runtime.",
            retryable=True,
        )
    if not callable(getattr(transformers_hub, "is_offline_mode", None)):
        raise WorkerError(
            "TRAINING_RUNTIME_TRANSFORMERS_INVALID",
            "O módulo de integração do Transformers não está íntegro.",
            retryable=True,
        )

    training_script = None
    entrypoint = None
    audio_probe = None
    if diffsynth_root is not None:
        training_script = diffsynth_root / "examples" / "wanvideo" / "model_training" / "train.py"
        if not training_script.is_file():
            raise WorkerError(
                "TRAINING_RUNTIME_SCRIPT_MISSING",
                "O script oficial do treinamento não foi encontrado.",
                retryable=True,
            )
        module = _load_training_entrypoint(training_script)
        entrypoint = {
            "wanParser": callable(getattr(module, "wan_parser", None)),
            "trainingModule": isinstance(getattr(module, "WanTrainingModule", None), type),
        }
        audio_probe = _probe_audio_operator()
        model_loader_contract = _probe_model_loader_contract()
    else:
        model_loader_contract = None

    return {
        "status": "IDENTITY_LORA_TRAINING_RUNTIME_READY",
        "versions": versions,
        "trainingScript": str(training_script) if training_script else None,
        "entrypoint": entrypoint,
        "audioProbe": audio_probe,
        "modelLoaderContract": model_loader_contract,
    }


def assert_runtime_compatible(diffsynth_root: Path) -> dict[str, Any]:
    return inspect_runtime(diffsynth_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida o runtime completo do trainer sem iniciar GPU ou treinamento.")
    parser.add_argument("--diffsynth-root", default="/opt/DiffSynth-Studio")
    args = parser.parse_args()
    print(json.dumps(inspect_runtime(Path(args.diffsynth_root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
