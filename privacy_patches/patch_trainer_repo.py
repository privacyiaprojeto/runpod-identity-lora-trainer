#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

PATCH_ID = "D3_6H13_RAM_CACHE_CHECKPOINT_RESCUE_V1"
BEGIN = f"# BEGIN {PATCH_ID}"
END = f"# END {PATCH_ID}"

IMPORT_SAFE_PATCH_ID = "D3_6H14_1_IMPORT_SAFE_RUNTIME_PREFLIGHT_V1"
IMPORT_SAFE_BEGIN = f"# BEGIN {IMPORT_SAFE_PATCH_ID}"
IMPORT_SAFE_END = f"# END {IMPORT_SAFE_PATCH_ID}"

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

POST_OVERLAY_PREFLIGHT_BLOCK = r"""
# BEGIN D3_6H14_1_IMPORT_SAFE_RUNTIME_PREFLIGHT_V1
# Validate the exact overlay installed in the final image, not only upstream train.py.
RUN python -m compileall -q /app \
    && python -m py_compile \
        /opt/DiffSynth-Studio/examples/wanvideo/model_training/train.py \
        /opt/DiffSynth-Studio/examples/wanvideo/model_training/train.privacy_original.py \
    && python -m identity_worker.runtime_preflight --diffsynth-root /opt/DiffSynth-Studio
# END D3_6H14_1_IMPORT_SAFE_RUNTIME_PREFLIGHT_V1
""".strip()


def append_block_if_missing(path: Path, marker: str, block: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    path.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8", newline="\n")
    return True


def insert_docker_block(path: Path) -> bool:
    return append_block_if_missing(path, BEGIN, DOCKER_BLOCK)


def insert_post_overlay_preflight(path: Path) -> bool:
    changed = append_block_if_missing(path, IMPORT_SAFE_BEGIN, POST_OVERLAY_PREFLIGHT_BLOCK)
    text = path.read_text(encoding="utf-8")
    if text.index(IMPORT_SAFE_BEGIN) < text.index(BEGIN):
        raise SystemExit("Post-overlay preflight must appear after the RAM-cache overlay block")
    return changed


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
    post_overlay_preflight_changed = insert_post_overlay_preflight(dockerfile)

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
        "status": "D3_6H14_1_TRAINER_REPO_PATCHED",
        "dockerfile_changed": docker_changed,
        "post_overlay_preflight_changed": post_overlay_preflight_changed,
        "bf16_launch": bf16_result,
        "bf16_file": str(bf16_path.relative_to(repo)),
        "save_steps": 400,
        "ram_cache_overlay": True,
        "import_safe_runtime_preflight": True,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
