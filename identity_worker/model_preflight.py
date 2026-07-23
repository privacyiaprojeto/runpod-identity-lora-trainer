from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .errors import WorkerError
from .model_lock import MaterializedModelBinding

_EXPECTED_MODEL_NAME = "wan_video_vace"


def _assert_files(binding: MaterializedModelBinding) -> None:
    if len(binding.diffusion_shards) != 7:
        raise WorkerError(
            "MODEL_BINDING_SHARDS_INVALID",
            "O binding do VACE não contém os sete shards agrupados.",
            retryable=False,
        )
    for value in (*binding.diffusion_shards, binding.text_encoder_path, binding.vae_path):
        path = Path(value)
        if not path.is_file() or path.stat().st_size <= 0:
            raise WorkerError(
                "MODEL_BINDING_FILE_MISSING",
                "Um artefato congelado do modelo-base não está disponível no volume privado.",
                retryable=True,
            )


def assert_model_binding_compatible(binding: MaterializedModelBinding) -> dict[str, Any]:
    _assert_files(binding)
    try:
        loader = importlib.import_module("diffsynth.core.loader")
        configs = importlib.import_module("diffsynth.configs")
        hash_model_file = getattr(loader, "hash_model_file")
        model_configs = list(getattr(configs, "MODEL_CONFIGS"))
        model_hash = str(hash_model_file(list(binding.diffusion_shards)))
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError(
            "TRAINING_MODEL_PREFLIGHT_FAILED",
            "O loader do DiffSynth não conseguiu inspecionar o checkpoint agrupado.",
            retryable=True,
        ) from exc

    def config_value(item: Any, key: str) -> Any:
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    matches = [
        item for item in model_configs
        if str(config_value(item, "model_hash") or "") == model_hash
    ]
    model_names = sorted({
        str(config_value(item, "model_name") or "")
        for item in matches
        if config_value(item, "model_name")
    })
    if _EXPECTED_MODEL_NAME not in model_names:
        raise WorkerError(
            "TRAINING_MODEL_DETECTION_FAILED",
            "O DiffSynth não reconheceu o checkpoint agrupado como Wan VACE homologado.",
            retryable=True,
        )

    return {
        "status": "IDENTITY_LORA_MODEL_BINDING_READY",
        "repository": binding.repository,
        "revision": binding.revision,
        "modelHash": model_hash,
        "modelName": _EXPECTED_MODEL_NAME,
        "diffusionShardCount": len(binding.diffusion_shards),
        "groupedModelPaths": True,
        "downloadAttemptedByPreflight": False,
        "weightsLoadedByPreflight": False,
    }
