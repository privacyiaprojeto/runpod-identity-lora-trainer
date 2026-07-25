from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "patch_diffsynth_runner.py"
SPEC = importlib.util.spec_from_file_location("patch_diffsynth_runner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

OFFICIAL_TAIL_FIXTURE = '''import os, json, torch, importlib
from tqdm import tqdm

def launch_training_task(accelerator, dataset, model, model_logger, save_steps=None, num_epochs=1, enable_model_cpu_offload=False):
    optimizer = object()
    scheduler = object()
    dataloader = object()
    initialize_deepspeed_gradient_checkpointing(accelerator)
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
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def initialize_deepspeed_gradient_checkpointing(accelerator):
    return None
'''


def test_patches_official_fb337fb_training_tail_structurally():
    patched, changed = MODULE.patch_runner_text(OFFICIAL_TAIL_FIXTURE)
    assert changed is True
    assert MODULE.MARKER in patched
    assert 'privacy_max_steps != 800' in patched
    assert 'model_logger.save_model(accelerator, model, f"step-{model_logger.num_steps}.safetensors")' in patched
    compile(patched, "runner.py", "exec")


def test_tolerates_formatting_that_broke_literal_replacement():
    formatted = OFFICIAL_TAIL_FIXTURE.replace(
        "    for epoch_id in range(num_epochs):\n",
        "    # upstream formatting/comment preserved\n    for epoch_id in range(num_epochs):   \n",
    )
    patched, changed = MODULE.patch_runner_text(formatted)
    assert changed is True
    assert patched.count(MODULE.MARKER) == 1


def test_is_idempotent_after_patch():
    patched, changed = MODULE.patch_runner_text(OFFICIAL_TAIL_FIXTURE)
    assert changed is True
    second, second_changed = MODULE.patch_runner_text(patched)
    assert second_changed is False
    assert second == patched


def test_blocks_semantic_runner_divergence():
    divergent = OFFICIAL_TAIL_FIXTURE.replace("                optimizer.step()\n", "")
    with pytest.raises(ValueError, match="optimizer_or_logger_calls"):
        MODULE.patch_runner_text(divergent)
