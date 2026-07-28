from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from identity_worker import runtime_preflight
from identity_worker.errors import WorkerError

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "privacy_patches" / "diffsynth_ram_cache_entrypoint.py"
DOCKERFILE = ROOT / "Dockerfile"
PATCHER = ROOT / "privacy_patches" / "patch_trainer_repo.py"


def _write_fixture_overlay(tmp_path: Path) -> tuple[Path, Path]:
    overlay = tmp_path / "train.py"
    original = tmp_path / "train.privacy_original.py"
    overlay.write_text(OVERLAY.read_text(encoding="utf-8"), encoding="utf-8")
    original.write_text(
        "class WanTrainingModule:\n"
        "    pass\n\n"
        "def wan_parser():\n"
        "    return 'fixture-parser'\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(91)\n",
        encoding="utf-8",
    )
    return overlay, original


def test_overlay_import_is_safe_and_exports_upstream_contract(tmp_path: Path, monkeypatch):
    overlay, _ = _write_fixture_overlay(tmp_path)
    argv_before = ["worker", "--unexpected", "value"]
    monkeypatch.setattr(sys, "argv", list(argv_before))

    spec = importlib.util.spec_from_file_location("privacy_overlay_preflight_fixture", overlay)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.wan_parser() == "fixture-parser"
    assert isinstance(module.WanTrainingModule, type)
    assert sys.argv == argv_before
    assert module.IMPORT_SAFE_PATCH_ID == "D3_6H14_1_IMPORT_SAFE_RUNTIME_PREFLIGHT_V1"


def test_overlay_cli_still_executes_original_main(tmp_path: Path, monkeypatch):
    overlay = tmp_path / "train.py"
    original = tmp_path / "train.privacy_original.py"
    marker = tmp_path / "cli-ran.txt"
    overlay.write_text(OVERLAY.read_text(encoding="utf-8"), encoding="utf-8")
    original.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text("
        "'save=' + sys.argv[sys.argv.index('--save_steps') + 1] + "
        "';bf16=' + os.environ.get('ACCELERATE_MIXED_PRECISION', ''), "
        "encoding='utf-8')\n",
        encoding="utf-8",
    )

    class FakeDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"index": index}

    accelerate = ModuleType("accelerate")
    accelerate.Accelerator = lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    diffsynth = ModuleType("diffsynth")
    core = ModuleType("diffsynth.core")
    core.UnifiedDataset = FakeDataset
    diffsynth.core = core
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "diffsynth", diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.core", core)
    monkeypatch.setattr(sys, "argv", [str(overlay)])
    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    runpy.run_path(str(overlay), run_name="__main__")

    assert marker.read_text(encoding="utf-8") == "save=400;bf16=bf16"


def test_runtime_preflight_converts_system_exit_to_worker_error(tmp_path: Path):
    training_script = tmp_path / "train.py"
    training_script.write_text("raise SystemExit(2)\n", encoding="utf-8")

    with pytest.raises(WorkerError) as captured:
        runtime_preflight._load_training_entrypoint(training_script)

    assert captured.value.code == "TRAINING_RUNTIME_ENTRYPOINT_SYSTEM_EXIT"
    assert captured.value.retryable is True
    assert "código 2" in str(captured.value)


def test_docker_validates_exact_overlay_after_installation():
    text = DOCKERFILE.read_text(encoding="utf-8")
    overlay_index = text.index("# BEGIN D3_6H13_RAM_CACHE_CHECKPOINT_RESCUE_V1")
    post_index = text.index("# BEGIN D3_6H14_1_IMPORT_SAFE_RUNTIME_PREFLIGHT_V1")
    assert post_index > overlay_index
    post_block = text[post_index:]
    assert "python -m identity_worker.runtime_preflight" in post_block
    assert "train.privacy_original.py" in post_block


def test_repo_patcher_contains_idempotent_post_overlay_guard():
    text = PATCHER.read_text(encoding="utf-8")
    assert "D3_6H14_1_IMPORT_SAFE_RUNTIME_PREFLIGHT_V1" in text
    assert "insert_post_overlay_preflight" in text
    assert "Post-overlay preflight must appear after" in text
