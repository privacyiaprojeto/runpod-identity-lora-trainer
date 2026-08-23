from __future__ import annotations

import json
import math
import os
import subprocess
from collections import deque
from pathlib import Path

from .adapter_scope import REQUIRED_CHECKPOINT_STEPS, collect_and_audit_checkpoints
from .errors import WorkerError
from .hashing import sha256_file
from .model_lock import MaterializedModelBinding

_AUDIO_DEPENDENCY_FAILURE_MARKERS = (
    "no module named 'librosa'", 'no module named "librosa"', "no module named 'soundfile'",
    'no module named "soundfile"', "no module named 'soxr'", 'no module named "soxr"',
    "no module named 'numba'", 'no module named "numba"', "no module named 'scipy'",
    'no module named "scipy"', "training_runtime_audio_dependency_missing",
)
_MODEL_DETECTION_FAILURE_MARKERS = ("cannot detect the model type", "training_model_detection_failed", "model_binding_shards_invalid", "model_binding_component_missing")
_MODEL_PREFLIGHT_FAILURE_MARKERS = ("training_model_preflight_failed", "training_model_loader_contract_invalid", "training_model_registry_missing_vace")
_IMPORT_FAILURE_MARKERS = ("importerror:", "modulenotfounderror:", "cannot import name")
_GPU_OOM_MARKERS = ("cuda out of memory", "outofmemoryerror")


def build_command(request, settings, dataset_root: Path, metadata_path: Path, model_binding: MaterializedModelBinding, output_dir: Path) -> list[str]:
    t = request.payload["training"]
    grouped_model_paths = model_binding.diffsynth_model_paths()
    return [
        "accelerate", "launch", '--num_processes', '1', '--mixed_precision', 'bf16', str(settings.diffsynth_root / "examples/wanvideo/model_training/train.py"),
        "--dataset_base_path", str(dataset_root), "--dataset_metadata_path", str(metadata_path),
        "--data_file_keys", "video,vace_video,vace_reference_image", "--height", str(t["height"]), "--width", str(t["width"]),
        "--num_frames", str(t["num_frames"]), "--dataset_repeat", str(t["dataset_repeat"]),
        "--model_paths", json.dumps(grouped_model_paths), "--learning_rate", str(t["learning_rate"]),
        "--num_epochs", str(t["num_epochs"]), "--save_steps", "400",
        "--remove_prefix_in_ckpt", "pipe.dit.", "--output_path", str(output_dir),
        "--lora_base_model", "dit", "--lora_target_modules", ",".join(t["target_modules"]),
        "--lora_rank", str(t["lora_rank"]), "--extra_inputs", "vace_video,vace_reference_image",
        "--use_gradient_checkpointing_offload",
    ]


# POD7C_H2_TRAINING_TELEMETRY_V1
TRAINING_TELEMETRY_SCHEMA = (
    "privacy-identity-training-telemetry-v1"
)

TRAINING_TELEMETRY_FILENAME = (
    "training_telemetry.jsonl"
)


def training_telemetry_path(
    output_dir: Path,
) -> Path:

    return (
        output_dir /
        "audit" /
        TRAINING_TELEMETRY_FILENAME
    )


def load_training_telemetry(
    output_dir: Path,
) -> dict:

    path = training_telemetry_path(
        output_dir
    )

    if not path.is_file():
        raise WorkerError(
            "TRAINING_TELEMETRY_MISSING",
            (
                "Arquivo de telemetria "
                "do treinamento ausente."
            ),
            retryable=False,
        )

    raw_lines = [
        line
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    expected_last_step = int(
        REQUIRED_CHECKPOINT_STEPS[-1]
    )

    if (
        len(raw_lines) !=
        expected_last_step + 1
    ):
        raise WorkerError(
            "TRAINING_TELEMETRY_COUNT_INVALID",
            (
                "Telemetria deve conter "
                f"{expected_last_step} steps "
                "mais o recibo terminal."
            ),
            retryable=False,
        )

    records = []

    for expected_step in range(
        1,
        expected_last_step + 1,
    ):

        try:
            record = json.loads(
                raw_lines[
                    expected_step - 1
                ]
            )

        except Exception as exc:
            raise WorkerError(
                "TRAINING_TELEMETRY_JSON_INVALID",
                (
                    "Registro JSON de "
                    "telemetria invalido."
                ),
                retryable=False,
            ) from exc

        if (
            record.get("event") !=
            "optimizer_step"
        ):
            raise WorkerError(
                "TRAINING_TELEMETRY_EVENT_INVALID",
                (
                    "Evento de step "
                    "de telemetria invalido."
                ),
                retryable=False,
            )

        if (
            int(
                record.get(
                    "step"
                ) or 0
            ) !=
            expected_step
        ):
            raise WorkerError(
                "TRAINING_TELEMETRY_STEP_INVALID",
                (
                    "Sequencia de optimizer "
                    "steps invalida."
                ),
                retryable=False,
            )

        try:
            loss = float(
                record["loss"]
            )

            learning_rate = float(
                record[
                    "learning_rate"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise WorkerError(
                "TRAINING_TELEMETRY_VALUE_INVALID",
                (
                    "Loss ou learning rate "
                    "ausente/invalido."
                ),
                retryable=False,
            ) from exc

        if (
            not math.isfinite(loss)
            or
            not math.isfinite(
                learning_rate
            )
            or
            learning_rate < 0
        ):
            raise WorkerError(
                "TRAINING_TELEMETRY_NON_FINITE",
                (
                    "Loss ou learning rate "
                    "nao finito."
                ),
                retryable=False,
            )

        expected_checkpoint = (
            expected_step
            in REQUIRED_CHECKPOINT_STEPS
        )

        if (
            bool(
                record.get(
                    "checkpoint"
                )
            )
            !=
            expected_checkpoint
        ):
            raise WorkerError(
                "TRAINING_TELEMETRY_CHECKPOINT_INVALID",
                (
                    "Marcacao de checkpoint "
                    "na telemetria divergiu."
                ),
                retryable=False,
            )

        records.append(
            {
                "step":
                    expected_step,

                "loss":
                    loss,

                "learning_rate":
                    learning_rate,

                "checkpoint":
                    expected_checkpoint,
            }
        )

    try:
        terminal = json.loads(
            raw_lines[-1]
        )

    except Exception as exc:
        raise WorkerError(
            "TRAINING_TELEMETRY_TERMINAL_INVALID",
            (
                "Recibo terminal de "
                "telemetria invalido."
            ),
            retryable=False,
        ) from exc

    if (
        terminal.get("event") !=
        "training_end"
        or
        int(
            terminal.get(
                "step"
            ) or 0
        )
        !=
        expected_last_step
        or
        terminal.get(
            "termination_reason"
        )
        !=
        "optimizer_step_contract_reached"
    ):
        raise WorkerError(
            "TRAINING_TELEMETRY_TERMINATION_INVALID",
            (
                "Treinamento nao terminou "
                "pelo contrato exato de steps."
            ),
            retryable=False,
        )

    try:
        elapsed_seconds = float(
            terminal[
                "elapsed_seconds"
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise WorkerError(
            "TRAINING_TELEMETRY_ELAPSED_INVALID",
            (
                "Elapsed time terminal "
                "ausente/invalido."
            ),
            retryable=False,
        ) from exc

    if (
        not math.isfinite(
            elapsed_seconds
        )
        or
        elapsed_seconds < 0
    ):
        raise WorkerError(
            "TRAINING_TELEMETRY_ELAPSED_INVALID",
            (
                "Elapsed time terminal "
                "nao finito."
            ),
            retryable=False,
        )

    losses = [
        record["loss"]
        for record in records
    ]

    rates = [
        record[
            "learning_rate"
        ]
        for record in records
    ]

    return {
        "schema_version":
            TRAINING_TELEMETRY_SCHEMA,

        "path":
            path,

        "sha256":
            sha256_file(
                path
            ),

        "bytes":
            path.stat().st_size,

        "step_count":
            len(
                records
            ),

        "first_step":
            records[0][
                "step"
            ],

        "last_step":
            records[-1][
                "step"
            ],

        "checkpoint_steps":
            list(
                REQUIRED_CHECKPOINT_STEPS
            ),

        "loss_first":
            losses[0],

        "loss_last":
            losses[-1],

        "loss_min":
            min(
                losses
            ),

        "loss_max":
            max(
                losses
            ),

        "learning_rate_first":
            rates[0],

        "learning_rate_last":
            rates[-1],

        "elapsed_seconds":
            elapsed_seconds,

        "termination_reason":
            terminal[
                "termination_reason"
            ],
    }


def _classify_failure(output_tail: str, return_code: int) -> WorkerError:
    normalized = output_tail.lower()
    if any(marker in normalized for marker in _AUDIO_DEPENDENCY_FAILURE_MARKERS):
        return WorkerError("TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING", "O treinamento não iniciou porque uma dependência interna de áudio do DiffSynth está ausente.", retryable=True)
    if any(marker in normalized for marker in _MODEL_DETECTION_FAILURE_MARKERS):
        return WorkerError("TRAINING_MODEL_DETECTION_FAILED", "O treinamento não iniciou porque o loader não reconheceu o modelo-base agrupado.", retryable=True)
    if any(marker in normalized for marker in _MODEL_PREFLIGHT_FAILURE_MARKERS):
        return WorkerError("TRAINING_MODEL_PREFLIGHT_FAILED", "O treinamento não iniciou porque o preflight do loader do modelo-base falhou.", retryable=True)
    if any(marker in normalized for marker in _IMPORT_FAILURE_MARKERS):
        return WorkerError("TRAINING_RUNTIME_IMPORT_FAILED", "O treinamento não iniciou porque o runtime Python está incompatível.", retryable=True)
    if any(marker in normalized for marker in _GPU_OOM_MARKERS):
        return WorkerError("TRAINING_GPU_OUT_OF_MEMORY", "O treinamento foi interrompido por memória insuficiente na GPU.", retryable=True)
    return WorkerError("DIFFSYNTH_TRAINING_FAILED", f"O treinamento encerrou com código {return_code}.", retryable=True)


def run_training(command: list[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = (
        training_telemetry_path(
            output_dir
        )
    )

    telemetry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if telemetry_path.exists():
        telemetry_path.unlink()

    tail: deque[str] = deque(
        maxlen=160
    )

    env = {
        **os.environ,

        "PRIVACY_MAX_OPTIMIZER_STEPS":
            "800",

        "PRIVACY_CHECKPOINT_STEPS":
            ",".join(
                str(step)
                for step
                in REQUIRED_CHECKPOINT_STEPS
            ),

        "PRIVACY_TRAINING_TELEMETRY_PATH":
            str(
                telemetry_path
            ),
    }

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(
            line,
            end="",
            flush=True,
        )

        tail.append(
            line
        )

    return_code = (
        process.wait()
    )

    if return_code != 0:
        raise _classify_failure(
            "".join(
                tail
            ),
            return_code,
        )

    # Fail closed before accepting checkpoints.
    load_training_telemetry(
        output_dir
    )

    checkpoints = (
        collect_and_audit_checkpoints(
            output_dir
        )
    )

    return checkpoints[-1][
        "path"
    ]
