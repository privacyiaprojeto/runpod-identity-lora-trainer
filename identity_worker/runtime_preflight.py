from __future__ import annotations

import argparse
import importlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any

from .errors import WorkerError

EXPECTED_VERSIONS = {
    "transformers": "4.56.2",
    "huggingface_hub": "0.35.1",
    "accelerate": "1.10.1",
    "tokenizers": "0.22.1",
    "peft": "0.17.1",
}


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise WorkerError(
            "TRAINING_RUNTIME_DEPENDENCY_MISSING",
            f"Dependência obrigatória ausente: {distribution}.",
            retryable=True,
        ) from exc


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

    try:
        importlib.import_module("transformers")
        accelerate_cli = importlib.import_module("accelerate.commands.accelerate_cli")
        transformers_hub = importlib.import_module("transformers.utils.hub")
    except (ImportError, ModuleNotFoundError) as exc:
        raise WorkerError(
            "TRAINING_RUNTIME_IMPORT_FAILED",
            "O runtime do treinamento possui dependências Python incompatíveis.",
            retryable=True,
        ) from exc

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
    if diffsynth_root is not None:
        training_script = diffsynth_root / "examples" / "wanvideo" / "model_training" / "train.py"
        if not training_script.is_file():
            raise WorkerError(
                "TRAINING_RUNTIME_SCRIPT_MISSING",
                "O script oficial do treinamento não foi encontrado.",
                retryable=True,
            )

    return {
        "status": "IDENTITY_LORA_TRAINING_RUNTIME_READY",
        "versions": versions,
        "trainingScript": str(training_script) if training_script else None,
    }


def assert_runtime_compatible(diffsynth_root: Path) -> dict[str, Any]:
    return inspect_runtime(diffsynth_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida o runtime do trainer sem iniciar GPU ou treinamento.")
    parser.add_argument("--diffsynth-root", default="/opt/DiffSynth-Studio")
    args = parser.parse_args()
    print(json.dumps(inspect_runtime(Path(args.diffsynth_root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
