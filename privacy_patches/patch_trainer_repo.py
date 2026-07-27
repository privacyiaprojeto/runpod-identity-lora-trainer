#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

PATCH_ID = "D3_6H13_RAM_CACHE_CHECKPOINT_RESCUE_V1"
BEGIN = f"# BEGIN {PATCH_ID}"
END = f"# END {PATCH_ID}"

DOCKER_BLOCK = r"""
# BEGIN D3_6H13_RAM_CACHE_CHECKPOINT_RESCUE_V1
# Preserve upstream Wan entrypoint and replace it with the Privacy IA RAM-cache overlay.
COPY privacy_patches/diffsynth_ram_cache_entrypoint.py /opt/DiffSynth-Studio/examples/wanvideo/model_training/train.privacy_ram_cache.py
RUN set -eux; \
    original=/opt/DiffSynth-Studio/examples/wanvideo/model_training/train.py; \
    backup=/opt/DiffSynth-Studio/examples/wanvideo/model_training/train.privacy_original.py; \
    test -f "$original"; \
    if [ ! -f "$backup" ]; then cp "$original" "$backup"; fi; \
    cp /opt/DiffSynth-Studio/examples/wanvideo/model_training/train.privacy_ram_cache.py "$original"
# END D3_6H13_RAM_CACHE_CHECKPOINT_RESCUE_V1
""".strip()


def insert_docker_block(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if BEGIN in text:
        return False
    path.write_text(text.rstrip() + "\n\n" + DOCKER_BLOCK + "\n", encoding="utf-8", newline="\n")
    return True


def patch_accelerate_source(path: Path) -> tuple[bool, str]:
    text = path.read_text(encoding="utf-8")
    if "--mixed_precision" in text and "bf16" in text:
        return False, "already_present"

    patterns = [
        (
            re.compile(r"((?:['\"]-m['\"]\s*,\s*)?['\"]accelerate\.commands\.launch['\"]\s*,)"),
            r"\1 '--mixed_precision', 'bf16',",
        ),
        (
            re.compile(r"(['\"]accelerate['\"]\s*,\s*['\"]launch['\"]\s*,)"),
            r"\1 '--mixed_precision', 'bf16',",
        ),
    ]
    for pattern, replacement in patterns:
        updated, count = pattern.subn(replacement, text, count=1)
        if count == 1:
            path.write_text(updated, encoding="utf-8", newline="\n")
            return True, "patched"
    return False, "pattern_not_found"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    dockerfile = repo / "Dockerfile"
    if not dockerfile.is_file():
        raise SystemExit(f"Dockerfile not found: {dockerfile}")

    docker_changed = insert_docker_block(dockerfile)

    candidates = [repo / "identity_worker" / "trainer.py"]
    candidates.extend(sorted((repo / "identity_worker").glob("*.py")))
    seen = set()
    bf16_result = None
    bf16_path = None
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        source = candidate.read_text(encoding="utf-8", errors="replace")
        if "accelerate" not in source:
            continue
        changed, result = patch_accelerate_source(candidate)
        if changed or result == "already_present":
            bf16_result = result
            bf16_path = candidate
            break

    if bf16_result is None:
        raise SystemExit("Could not locate the accelerate launch command; patch aborted fail-closed")

    report = {
        "status": "D3_6H13_TRAINER_REPO_PATCHED",
        "dockerfile_changed": docker_changed,
        "bf16_launch": bf16_result,
        "bf16_file": str(bf16_path.relative_to(repo)),
        "save_steps": 400,
        "ram_cache_overlay": True,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
