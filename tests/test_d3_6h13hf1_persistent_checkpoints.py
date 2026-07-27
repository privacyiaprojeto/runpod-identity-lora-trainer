from pathlib import Path
from types import SimpleNamespace

import pytest

from identity_worker.errors import WorkerError
from identity_worker.run_paths import (
    assert_persistent_runtime_root,
    prepare_training_output_dir,
    training_output_dir,
)
from identity_worker.trainer import build_command


ACTOR_ID = "767f0277-6b78-4afb-8b5e-88b302033705"
RUN_ID = "f305faa7-a687-4c16-a5c0-bc3b89ae2c28"


def test_training_output_is_actor_and_run_scoped(tmp_path):
    runtime_root = tmp_path / "runpod-volume" / "privacy-identity-lora"
    output = training_output_dir(runtime_root, ACTOR_ID, RUN_ID)
    expected = runtime_root.resolve() / "training-runs" / ACTOR_ID / RUN_ID / "output"
    assert output == expected


def test_nonempty_training_output_is_fail_closed(tmp_path):
    runtime_root = tmp_path / "runpod-volume" / "privacy-identity-lora"
    output = prepare_training_output_dir(runtime_root, ACTOR_ID, RUN_ID)
    (output / "step-400.safetensors").write_bytes(b"existing")
    with pytest.raises(WorkerError) as error:
        prepare_training_output_dir(runtime_root, ACTOR_ID, RUN_ID)
    assert error.value.code == "TRAINING_OUTPUT_ALREADY_EXISTS"


def test_mount_proof_accepts_runpod_volume(tmp_path):
    runtime_root = Path("/runpod-volume/privacy-identity-lora")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "36 25 0:44 / /runpod-volume rw,relatime - nfs server:/volume rw\n",
        encoding="utf-8",
    )
    result = assert_persistent_runtime_root(runtime_root, mountinfo_path=mountinfo)
    assert result["mount_point"] == "/runpod-volume"


def test_mount_proof_rejects_plain_root_filesystem(tmp_path):
    runtime_root = Path("/runpod-volume/privacy-identity-lora")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "25 1 0:1 / / rw,relatime - overlay overlay rw\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkerError) as error:
        assert_persistent_runtime_root(runtime_root, mountinfo_path=mountinfo)
    assert error.value.code == "TRAINING_NETWORK_VOLUME_NOT_MOUNTED"


def test_mount_proof_rejects_runtime_outside_runpod_volume(tmp_path):
    runtime_root = Path("/tmp/privacy-identity-lora")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "36 25 0:44 / /runpod-volume rw,relatime - nfs server:/volume rw\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkerError) as error:
        assert_persistent_runtime_root(runtime_root, mountinfo_path=mountinfo)
    assert error.value.code == "TRAINING_RUNTIME_ROOT_NOT_PERSISTENT"


def test_training_command_uses_bf16_and_400(tmp_path):
    request = SimpleNamespace(payload={"training": {
        "height": 480,
        "width": 832,
        "num_frames": 17,
        "dataset_repeat": 60,
        "learning_rate": 0.00005,
        "num_epochs": 1,
        "lora_rank": 32,
        "target_modules": [
            "cross_attn.q", "cross_attn.k", "cross_attn.v",
            "cross_attn.o", "ffn.0", "ffn.2",
        ],
    }})
    settings = SimpleNamespace(diffsynth_root=Path("/opt/DiffSynth-Studio"))
    binding = SimpleNamespace(
        diffsynth_model_paths=lambda: [
            ["/models/shard.safetensors"],
            "/models/t5.pth",
            "/models/vae.pth",
        ]
    )
    command = build_command(
        request,
        settings,
        tmp_path,
        tmp_path / "metadata.csv",
        binding,
        tmp_path / "output",
    )
    joined = " ".join(command)
    assert "--mixed_precision bf16" in joined
    assert "--save_steps 400" in joined


def test_handler_keeps_checkpoint_output_outside_temporary_directory():
    handler = (Path(__file__).resolve().parents[1] / "handler.py").read_text(
        encoding="utf-8"
    )
    assert "prepare_training_output_dir(" in handler
    assert "assert_persistent_runtime_root(" in handler
    assert "TemporaryDirectory(prefix=f'identity_" in handler
    assert (
        "TemporaryDirectory(dir=str(settings.runtime_root), prefix=f'identity_"
        not in handler
    )
    assert "output_dir = work / 'output'" not in handler
