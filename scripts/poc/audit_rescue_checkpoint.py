#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

STEP_RE = re.compile(r"(?:step|checkpoint)[-_]?(\d+)", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extract_step(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = STEP_RE.search(part)
        if match:
            return int(match.group(1))
    return None


def inspect_safetensors(path: Path, full_tensor_read: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "step": extract_step(path),
        "safetensors_header_valid": False,
        "tensor_count": None,
        "full_tensor_read": False,
        "integrity": "unknown",
        "error": None,
    }
    if result["size_bytes"] <= 0:
        result["integrity"] = "invalid_empty"
        return result
    try:
        from safetensors import safe_open
    except Exception as exc:
        result["integrity"] = "sha256_only_library_missing"
        result["error"] = f"safetensors import failed: {exc}"
        return result

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            result["tensor_count"] = len(keys)
            result["safetensors_header_valid"] = len(keys) > 0
            if full_tensor_read:
                for key in keys:
                    tensor = handle.get_tensor(key)
                    _ = tensor.numel()
                result["full_tensor_read"] = True
        result["integrity"] = "valid" if result["safetensors_header_valid"] else "invalid_no_tensors"
    except Exception as exc:
        result["integrity"] = "invalid_safetensors"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def find_run_dirs(volume_root: Path, run_id: str) -> list[Path]:
    exact_prefix = f"identity_{run_id}_"
    candidates = []
    if volume_root.is_dir():
        for path in volume_root.rglob("*"):
            if path.is_dir() and (path.name == run_id or path.name.startswith(exact_prefix) or run_id in path.name):
                candidates.append(path)
    return sorted(set(candidates))


def choose(candidates: list[dict[str, Any]], expected: int, fallback: int) -> dict[str, Any] | None:
    valid = [item for item in candidates if item["integrity"] == "valid"]
    for step in (expected, fallback):
        exact = [item for item in valid if item["step"] == step]
        if exact:
            return sorted(exact, key=lambda item: item["size_bytes"], reverse=True)[0]
    lower = [item for item in valid if item["step"] is not None and item["step"] <= expected]
    if lower:
        return sorted(lower, key=lambda item: (item["step"], item["size_bytes"]), reverse=True)[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only audit and non-destructive checkpoint rescue")
    parser.add_argument("--volume-root", default="/runpod-volume/privacy-identity-lora")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-step", type=int, default=400)
    parser.add_argument("--fallback-step", type=int, default=200)
    parser.add_argument("--report", default="checkpoint-rescue-report.json")
    parser.add_argument("--copy-selected", default=None, help="Optional directory for a non-destructive copy")
    parser.add_argument("--full-tensor-read", action="store_true")
    args = parser.parse_args()

    root = Path(args.volume_root).resolve()
    run_dirs = find_run_dirs(root, args.run_id)
    files: list[Path] = []
    for run_dir in run_dirs:
        files.extend(run_dir.rglob("*.safetensors"))
    files = sorted(set(path.resolve() for path in files if path.is_file()))

    inspected = [inspect_safetensors(path, args.full_tensor_read) for path in files]
    selected = choose(inspected, args.expected_step, args.fallback_step)
    copied_to = None
    if selected and args.copy_selected:
        destination_dir = Path(args.copy_selected).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        source = Path(selected["path"])
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        if sha256_file(destination) != selected["sha256"]:
            destination.unlink(missing_ok=True)
            raise RuntimeError("COPY_HASH_MISMATCH")
        copied_to = str(destination)

    report = {
        "status": "CHECKPOINT_RESCUE_READY" if selected else "CHECKPOINT_RESCUE_NOT_FOUND",
        "run_id": args.run_id,
        "volume_root": str(root),
        "run_directories": [str(path) for path in run_dirs],
        "candidate_count": len(inspected),
        "candidates": inspected,
        "selected_checkpoint": selected,
        "copied_to": copied_to,
        "originals_modified": False,
        "r2_read_executed": False,
        "runpod_job_submitted": False,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
