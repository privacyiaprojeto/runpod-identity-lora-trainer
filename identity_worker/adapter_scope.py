from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import WorkerError

ALLOWED_TARGETS = (
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)
FORBIDDEN_TARGETS = ("vace", "vace_blocks", "self_attn")
REQUIRED_CHECKPOINT_STEPS = (400, 600, 800)


def _safetensors_keys(path: Path) -> tuple[str, ...]:
    try:
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little", signed=False)
            if header_size <= 2 or header_size > 64 * 1024 * 1024:
                raise ValueError("invalid header size")
            header = json.loads(handle.read(header_size).decode("utf-8"))
    except Exception as exc:
        raise WorkerError("INVALID_ADAPTER_SAFETENSORS", f"Checkpoint inválido: {path.name}") from exc
    keys = tuple(key for key in header if key != "__metadata__")
    if not keys:
        raise WorkerError("EMPTY_ADAPTER_SAFETENSORS", f"Checkpoint sem tensores: {path.name}")
    return keys


def _step(path: Path) -> int | None:
    match = re.search(r"step-(\d+)\.safetensors$", path.name, re.I)
    return int(match.group(1)) if match else None


def audit_checkpoint(path: Path) -> dict:
    keys = _safetensors_keys(path)
    unexpected = []
    for key in keys:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_TARGETS):
            unexpected.append(key)
        elif not any(token in lowered for token in ALLOWED_TARGETS):
            unexpected.append(key)
    if unexpected:
        raise WorkerError(
            "ADAPTER_TARGET_SCOPE_INVALID",
            f"O adapter contém tensores fora do escopo DiT aprovado em {path.name}: {unexpected[:20]}",
        )
    return {"path": path, "step": _step(path), "tensor_count": len(keys)}


def collect_and_audit_checkpoints(output_dir: Path) -> tuple[dict, ...]:
    found = {}
    for path in output_dir.rglob("step-*.safetensors"):
        step = _step(path)
        if step in REQUIRED_CHECKPOINT_STEPS:
            found[step] = audit_checkpoint(path)
    missing = [step for step in REQUIRED_CHECKPOINT_STEPS if step not in found]
    if missing:
        raise WorkerError("REQUIRED_CHECKPOINTS_MISSING", f"O treino não produziu todos os checkpoints controlados: {missing}")
    extras = [path.name for path in output_dir.rglob("step-*.safetensors") if _step(path) not in REQUIRED_CHECKPOINT_STEPS]
    if extras:
        raise WorkerError("UNEXPECTED_CHECKPOINTS_PRESENT", f"O treino produziu checkpoints fora do plano aprovado: {extras[:20]}")
    return tuple(found[step] for step in REQUIRED_CHECKPOINT_STEPS)
