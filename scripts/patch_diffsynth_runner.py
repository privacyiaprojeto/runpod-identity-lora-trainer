#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

MARKER = "PRIVACY_WAN_DIT_EXACT_STEP_PATCH_V1"
ORIGINAL = """    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
"""
PATCHED = """    # PRIVACY_WAN_DIT_EXACT_STEP_PATCH_V1
    privacy_max_steps = int(os.environ.get("PRIVACY_MAX_OPTIMIZER_STEPS", "0") or 0)
    privacy_checkpoint_steps = {
        int(item) for item in os.environ.get("PRIVACY_CHECKPOINT_STEPS", "").split(",") if item.strip()
    }
    if privacy_max_steps:
        if privacy_checkpoint_steps != {400, 600, 800} or privacy_max_steps != 800:
            raise ValueError("Invalid Privacy IA exact-step checkpoint contract.")
        save_steps = None
    privacy_stop = False
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                if privacy_max_steps and model_logger.num_steps in privacy_checkpoint_steps:
                    model_logger.save_model(accelerator, model, f"step-{model_logger.num_steps}.safetensors")
                if privacy_max_steps and model_logger.num_steps >= privacy_max_steps:
                    privacy_stop = True
            if privacy_stop:
                break
        if privacy_stop:
            break
        if save_steps is None and not privacy_max_steps:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    if privacy_max_steps and model_logger.num_steps != privacy_max_steps:
        raise RuntimeError(f"Exact optimizer step contract not reached: {model_logger.num_steps}/{privacy_max_steps}")
    model_logger.on_training_end(accelerator, model, save_steps)
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if not commit.startswith(args.expected_commit):
        raise SystemExit(f"BLOCKED_DIFFSYNTH_COMMIT: expected={args.expected_commit} actual={commit}")
    path = root / "diffsynth" / "diffusion" / "runner.py"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("DIFFSYNTH_EXACT_STEP_PATCH_ALREADY_APPLIED")
        return 0
    if text.count(ORIGINAL) != 1:
        raise SystemExit("BLOCKED_DIFFSYNTH_RUNNER_BASE_DIVERGED")
    path.write_text(text.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("DIFFSYNTH_EXACT_STEP_PATCH_APPLIED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
