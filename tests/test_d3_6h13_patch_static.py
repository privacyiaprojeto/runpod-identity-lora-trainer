from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def test_wrapper_contract_tokens():
    text = (ROOT / "privacy_patches" / "diffsynth_ram_cache_entrypoint.py").read_text(encoding="utf-8")
    required = [
        "EagerRamCachedUnifiedDataset",
        "privacy_ram_cache_ready",
        '_set_cli_value("--save_steps", "400")',
        'kwargs.setdefault("mixed_precision", "bf16")',
        "train.privacy_original.py",
        "reader.close()",
    ]
    for token in required:
        assert token in text


def test_checkpoint_auditor_imports():
    path = ROOT / "scripts" / "poc" / "audit_rescue_checkpoint.py"
    spec = importlib.util.spec_from_file_location("audit_rescue_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.extract_step(Path("/x/output/step-400.safetensors")) == 400


def test_repo_patcher_on_fixture():
    with tempfile.TemporaryDirectory() as temp:
        repo = Path(temp)
        (repo / "identity_worker").mkdir()
        (repo / "Dockerfile").write_text("FROM python:3.11\nCMD [\"python\", \"handler.py\"]\n", encoding="utf-8")
        (repo / "identity_worker" / "trainer.py").write_text(
            "def command(settings):\n"
            "    return ['python', '-m', 'accelerate.commands.launch', settings.training_script]\n",
            encoding="utf-8",
        )
        subprocess.run([
            sys.executable,
            str(ROOT / "privacy_patches" / "patch_trainer_repo.py"),
            "--repo", str(repo),
        ], check=True, capture_output=True, text=True)
        docker = (repo / "Dockerfile").read_text(encoding="utf-8")
        trainer = (repo / "identity_worker" / "trainer.py").read_text(encoding="utf-8")
        assert "D3_6H13_RAM_CACHE_CHECKPOINT_RESCUE_V1" in docker
        assert "--mixed_precision" in trainer and "bf16" in trainer
