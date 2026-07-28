#!/usr/bin/env python3
"""Privacy IA DiffSynth Wan entrypoint overlay.

Executed in place of DiffSynth-Studio's Wan train.py. During normal CLI
execution it installs the governed RAM-cache/BF16 patches and then delegates to
the preserved upstream entrypoint. During import-based runtime preflight it
loads only the upstream contract and never executes argparse or training.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
from types import ModuleType
from typing import Any


IMPORT_SAFE_PATCH_ID = "D3_6H14_1_IMPORT_SAFE_RUNTIME_PREFLIGHT_V1"


def _set_cli_value(flag: str, value: str, *, append_if_missing: bool = True) -> None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        if append_if_missing:
            sys.argv.extend([flag, value])
        return
    if index + 1 >= len(sys.argv):
        sys.argv.append(value)
    else:
        sys.argv[index + 1] = value


def _materialize(value: Any) -> Any:
    """Detach values from files/readers and keep reusable CPU-resident copies."""
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous().clone()
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.copy()
    except Exception:
        pass

    try:
        from PIL import Image
        if isinstance(value, Image.Image):
            value.load()
            detached = value.copy()
            try:
                value.close()
            except Exception:
                pass
            detached.load()
            return detached
    except Exception:
        pass

    if isinstance(value, dict):
        return {key: _materialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize(item) for item in value)
    return copy.deepcopy(value)


def _clone_from_ram(value: Any) -> Any:
    """Return mutable working copies without touching disk or FFmpeg."""
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.clone()
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.copy()
    except Exception:
        pass

    try:
        from PIL import Image
        if isinstance(value, Image.Image):
            return value.copy()
    except Exception:
        pass

    if isinstance(value, dict):
        return {key: _clone_from_ram(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_from_ram(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_from_ram(item) for item in value)
    return copy.deepcopy(value)


def _install_runtime_patches() -> None:
    import accelerate
    import diffsynth.core as core

    original_dataset = core.UnifiedDataset
    original_accelerator = accelerate.Accelerator

    class EagerRamCachedUnifiedDataset(original_dataset):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            started = time.monotonic()
            super().__init__(*args, **kwargs)
            logical_length = original_dataset.__len__(self)
            unique_length = len(self.cached_data) if self.load_from_cache else len(self.data)
            if unique_length < 1:
                raise RuntimeError("RAM_CACHE_EMPTY_DATASET: no unique samples were found")

            ram_cache = []
            decode_attempts = max(1, int(os.getenv("PRIVACY_RAM_CACHE_DECODE_ATTEMPTS", "2")))
            for item_index in range(unique_length):
                last_error = None
                for attempt in range(1, decode_attempts + 1):
                    active_readers = []
                    try:
                        import imageio
                        original_get_reader = imageio.get_reader

                        def tracked_get_reader(*reader_args: Any, **reader_kwargs: Any):
                            reader = original_get_reader(*reader_args, **reader_kwargs)
                            active_readers.append(reader)
                            return reader

                        imageio.get_reader = tracked_get_reader
                        item = original_dataset.__getitem__(self, item_index)
                        ram_cache.append(_materialize(item))
                        last_error = None
                        break
                    except OSError as exc:
                        last_error = exc
                        if attempt < decode_attempts:
                            time.sleep(0.5 * attempt)
                    finally:
                        try:
                            imageio.get_reader = original_get_reader
                        except Exception:
                            pass
                        for reader in active_readers:
                            try:
                                reader.close()
                            except Exception:
                                pass
                if last_error is not None:
                    raise RuntimeError(
                        f"RAM_CACHE_DECODE_FAILED: sample={item_index} attempts={decode_attempts}: {last_error}"
                    ) from last_error

            self._privacy_ram_cache = tuple(ram_cache)
            self._privacy_logical_length = logical_length
            elapsed = round(time.monotonic() - started, 3)
            print(json.dumps({
                "event": "privacy_ram_cache_ready",
                "unique_samples": unique_length,
                "logical_samples": logical_length,
                "preload_seconds": elapsed,
                "disk_reads_during_training_loop": 0,
            }, sort_keys=True), flush=True)

        def __len__(self) -> int:
            return self._privacy_logical_length

        def __getitem__(self, data_id: int) -> Any:
            item = self._privacy_ram_cache[data_id % len(self._privacy_ram_cache)]
            return _clone_from_ram(item)

    def bf16_accelerator(*args: Any, **kwargs: Any):
        kwargs.setdefault("mixed_precision", "bf16")
        return original_accelerator(*args, **kwargs)

    core.UnifiedDataset = EagerRamCachedUnifiedDataset
    accelerate.Accelerator = bf16_accelerator


def _original_entrypoint_path() -> Path:
    path = Path(__file__).with_name("train.privacy_original.py")
    if not path.is_file():
        raise RuntimeError(f"RAM_CACHE_ORIGINAL_ENTRYPOINT_MISSING: {path}")
    return path


def _load_original_contract() -> ModuleType:
    """Import the preserved upstream module without entering its CLI block."""
    original_entrypoint = _original_entrypoint_path()
    module_name = "privacy_identity_lora_wan_original_contract"
    spec = importlib.util.spec_from_file_location(module_name, original_entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"RAM_CACHE_ORIGINAL_ENTRYPOINT_LOADER_FAILED: {original_entrypoint}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module


def _run_cli() -> None:
    """Install runtime patches and delegate to the preserved upstream CLI."""
    # The upstream parser supports save_steps. Force the governed checkpoint cadence.
    _set_cli_value("--save_steps", "400")
    # If the command already exposes worker count, force a single-process in-memory cache.
    _set_cli_value("--dataset_num_workers", "0", append_if_missing=False)
    os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _install_runtime_patches()

    original_entrypoint = _original_entrypoint_path()
    sys.argv[0] = str(original_entrypoint)
    runpy.run_path(str(original_entrypoint), run_name="__main__")


if __name__ == "__main__":
    _run_cli()
else:
    # Runtime preflight imports this overlay. Export the same governed contract
    # as upstream while keeping argparse and training completely inactive.
    _upstream_contract = _load_original_contract()
    wan_parser = _upstream_contract.wan_parser
    WanTrainingModule = _upstream_contract.WanTrainingModule
    __all__ = ["wan_parser", "WanTrainingModule"]
