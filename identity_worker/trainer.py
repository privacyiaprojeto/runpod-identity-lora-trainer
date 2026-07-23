from __future__ import annotations

import json
import subprocess
from collections import deque
from pathlib import Path

from .errors import WorkerError
from .model_lock import MaterializedModelBinding

_AUDIO_DEPENDENCY_FAILURE_MARKERS = (
    "no module named 'librosa'",
    'no module named "librosa"',
    "no module named 'soundfile'",
    'no module named "soundfile"',
    "no module named 'soxr'",
    'no module named "soxr"',
    "no module named 'numba'",
    'no module named "numba"',
    "no module named 'scipy'",
    'no module named "scipy"',
    "training_runtime_audio_dependency_missing",
)
_MODEL_DETECTION_FAILURE_MARKERS = (
    "cannot detect the model type",
    "training_model_detection_failed",
    "model_binding_shards_invalid",
    "model_binding_component_missing",
)
_MODEL_PREFLIGHT_FAILURE_MARKERS = (
    "training_model_preflight_failed",
    "training_model_loader_contract_invalid",
    "training_model_registry_missing_vace",
)
_IMPORT_FAILURE_MARKERS = (
    "importerror:",
    "modulenotfounderror:",
    "cannot import name",
)
_GPU_OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
)


def build_command(
    request,
    settings,
    dataset_root: Path,
    metadata_path: Path,
    model_binding: MaterializedModelBinding,
    output_dir: Path,
) -> list[str]:
    t = request.payload["training"]
    grouped_model_paths = model_binding.diffsynth_model_paths()
    return [
        "accelerate", "launch", str(settings.diffsynth_root / "examples/wanvideo/model_training/train.py"),
        "--dataset_base_path", str(dataset_root), "--dataset_metadata_path", str(metadata_path),
        "--data_file_keys", "video,vace_video,vace_reference_image", "--height", str(t["height"]), "--width", str(t["width"]),
        "--num_frames", str(t["num_frames"]), "--dataset_repeat", str(t["dataset_repeat"]),
        "--model_paths", json.dumps(grouped_model_paths),
        "--learning_rate", str(t["learning_rate"]), "--num_epochs", str(t["num_epochs"]), "--remove_prefix_in_ckpt", "pipe.vace.",
        "--output_path", str(output_dir), "--lora_base_model", "vace", "--lora_target_modules", ",".join(t["target_modules"]),
        "--lora_rank", str(t["lora_rank"]), "--extra_inputs", "vace_video,vace_reference_image", "--use_gradient_checkpointing_offload",
    ]


def _classify_failure(output_tail: str, return_code: int) -> WorkerError:
    normalized = output_tail.lower()
    if any(marker in normalized for marker in _AUDIO_DEPENDENCY_FAILURE_MARKERS):
        return WorkerError(
            "TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING",
            "O treinamento não iniciou porque uma dependência interna de áudio do DiffSynth está ausente.",
            retryable=True,
        )
    if any(marker in normalized for marker in _MODEL_DETECTION_FAILURE_MARKERS):
        return WorkerError(
            "TRAINING_MODEL_DETECTION_FAILED",
            "O treinamento não iniciou porque o loader não reconheceu o modelo-base agrupado.",
            retryable=True,
        )
    if any(marker in normalized for marker in _MODEL_PREFLIGHT_FAILURE_MARKERS):
        return WorkerError(
            "TRAINING_MODEL_PREFLIGHT_FAILED",
            "O treinamento não iniciou porque o preflight do loader do modelo-base falhou.",
            retryable=True,
        )
    if any(marker in normalized for marker in _IMPORT_FAILURE_MARKERS):
        return WorkerError(
            "TRAINING_RUNTIME_IMPORT_FAILED",
            "O treinamento não iniciou porque o runtime Python está incompatível.",
            retryable=True,
        )
    if any(marker in normalized for marker in _GPU_OOM_MARKERS):
        return WorkerError(
            "TRAINING_GPU_OUT_OF_MEMORY",
            "O treinamento foi interrompido por memória insuficiente na GPU.",
            retryable=True,
        )
    return WorkerError(
        "DIFFSYNTH_TRAINING_FAILED",
        f"O treinamento encerrou com código {return_code}.",
        retryable=True,
    )


def run_training(command: list[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=160)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise _classify_failure("".join(tail), return_code)

    candidates = sorted(output_dir.rglob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise WorkerError(
            "ADAPTER_NOT_FOUND",
            "O treinamento terminou sem produzir o adapter esperado.",
            retryable=True,
        )
    return candidates[0]
