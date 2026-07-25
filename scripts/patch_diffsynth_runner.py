#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path

DOCKER_MARKER = "PRIVACY_WAN_DIT_EXACT_STEP_PATCH_V1"
MARKER = "PRIVACY_WAN_DIT_STRUCTURAL_PATCHER_V2"

PATCHED_TAIL = '''    # PRIVACY_WAN_DIT_EXACT_STEP_PATCH_V1
    # PRIVACY_WAN_DIT_STRUCTURAL_PATCHER_V2
    privacy_max_steps = int(os.environ.get("PRIVACY_MAX_OPTIMIZER_STEPS", "0") or 0)
    privacy_checkpoint_steps = {
        int(item) for item in os.environ.get("PRIVACY_CHECKPOINT_STEPS", "").split(",") if item.strip()
    }
    if privacy_max_steps:
        if privacy_checkpoint_steps != {400, 600, 800} or privacy_max_steps != 800:
            raise ValueError("Invalid Privacy IA exact-step checkpoint contract.")
        if int(getattr(accelerator, "gradient_accumulation_steps", 1)) != 1:
            raise ValueError("Privacy IA exact-step POC requires gradient_accumulation_steps=1.")
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
'''


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _is_named_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and _dotted_name(node.func) == name


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"BLOCKED_DIFFSYNTH_RUNNER_BASE_DIVERGED:{reason}")


def _find_training_tail(tree: ast.Module) -> tuple[int, int]:
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "launch_training_task"
    ]
    _require(len(functions) == 1, "launch_training_task")
    function = functions[0]
    _require(len(function.body) >= 2, "function_tail")

    outer_loop = function.body[-2]
    training_end = function.body[-1]

    _require(isinstance(outer_loop, ast.For), "outer_epoch_loop")
    _require(isinstance(outer_loop.target, ast.Name) and outer_loop.target.id == "epoch_id", "epoch_target")
    _require(
        _is_named_call(outer_loop.iter, "range")
        and len(outer_loop.iter.args) == 1
        and isinstance(outer_loop.iter.args[0], ast.Name)
        and outer_loop.iter.args[0].id == "num_epochs",
        "epoch_range",
    )
    _require(len(outer_loop.body) == 2, "epoch_body")

    data_loop = outer_loop.body[0]
    epoch_save = outer_loop.body[1]
    _require(isinstance(data_loop, ast.For), "data_loop")
    _require(isinstance(data_loop.target, ast.Name) and data_loop.target.id == "data", "data_target")
    _require(
        _is_named_call(data_loop.iter, "tqdm")
        and len(data_loop.iter.args) >= 1
        and isinstance(data_loop.iter.args[0], ast.Name)
        and data_loop.iter.args[0].id == "dataloader",
        "dataloader_tqdm",
    )
    _require(len(data_loop.body) == 1 and isinstance(data_loop.body[0], ast.With), "accumulate_with")

    accumulate = data_loop.body[0]
    _require(len(accumulate.items) == 1, "accumulate_context_count")
    context_expr = accumulate.items[0].context_expr
    _require(
        _is_named_call(context_expr, "accelerator.accumulate")
        and len(context_expr.args) == 1
        and isinstance(context_expr.args[0], ast.Name)
        and context_expr.args[0].id == "model",
        "accelerator_accumulate",
    )

    required_calls = [
        "accelerator.backward",
        "optimizer.step",
        "scheduler.step",
        "optimizer.zero_grad",
        "model_logger.on_step_end",
    ]
    call_lines: dict[str, int] = {}
    for node in ast.walk(accumulate):
        if isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name in required_calls and name not in call_lines:
                call_lines[name] = int(node.lineno)
    _require(all(name in call_lines for name in required_calls), "optimizer_or_logger_calls")
    _require(
        [call_lines[name] for name in required_calls]
        == sorted(call_lines[name] for name in required_calls),
        "optimizer_or_logger_order",
    )

    _require(isinstance(epoch_save, ast.If), "epoch_save_if")
    _require(
        isinstance(epoch_save.test, ast.Compare)
        and isinstance(epoch_save.test.left, ast.Name)
        and epoch_save.test.left.id == "save_steps"
        and len(epoch_save.test.ops) == 1
        and isinstance(epoch_save.test.ops[0], ast.Is)
        and len(epoch_save.test.comparators) == 1
        and isinstance(epoch_save.test.comparators[0], ast.Constant)
        and epoch_save.test.comparators[0].value is None,
        "epoch_save_condition",
    )
    _require(
        any(
            isinstance(node, ast.Call) and _dotted_name(node.func) == "model_logger.on_epoch_end"
            for node in ast.walk(epoch_save)
        ),
        "epoch_save_call",
    )

    _require(
        isinstance(training_end, ast.Expr)
        and _is_named_call(training_end.value, "model_logger.on_training_end"),
        "training_end_call",
    )
    _require(outer_loop.end_lineno is not None and training_end.end_lineno is not None, "line_ranges")
    return int(outer_loop.lineno), int(training_end.end_lineno)


def patch_runner_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False
    if DOCKER_MARKER in text:
        raise ValueError("BLOCKED_LEGACY_DIFFSYNTH_PATCH_PRESENT")
    tree = ast.parse(text)
    start_line, end_line = _find_training_tail(tree)
    lines = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"
    replacement = PATCHED_TAIL.replace("\n", newline)
    if not replacement.endswith(newline):
        replacement += newline
    patched = "".join(lines[: start_line - 1]) + replacement + "".join(lines[end_line:])
    compile(patched, "runner.py", "exec")
    if patched.count(MARKER) != 1:
        raise ValueError("BLOCKED_DIFFSYNTH_PATCH_MARKER_COUNT")
    return patched, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if not commit.startswith(args.expected_commit):
        raise SystemExit(
            f"BLOCKED_DIFFSYNTH_COMMIT: expected={args.expected_commit} actual={commit}"
        )

    path = root / "diffsynth" / "diffusion" / "runner.py"
    text = path.read_text(encoding="utf-8")
    try:
        patched, changed = patch_runner_text(text)
    except (SyntaxError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not changed:
        print("DIFFSYNTH_EXACT_STEP_PATCH_ALREADY_APPLIED")
        return 0
    path.write_text(patched, encoding="utf-8", newline="")
    print("DIFFSYNTH_EXACT_STEP_PATCH_APPLIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
