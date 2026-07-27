from pathlib import Path
from types import SimpleNamespace

from identity_worker.contracts import DIT_CHECKPOINT_STEPS, DIT_TARGET_MODULES, DIT_TRAINING_PROFILE
from identity_worker.trainer import build_command


def test_wan_dit_training_command_targets_generator_not_vace(tmp_path):
    request = SimpleNamespace(payload={"training": {
        "height": 480, "width": 832, "num_frames": 17, "dataset_repeat": 60,
        "learning_rate": 0.00005, "num_epochs": 1, "lora_rank": 32,
        "target_modules": list(DIT_TARGET_MODULES),
    }})
    settings = SimpleNamespace(diffsynth_root=Path("/opt/DiffSynth-Studio"))
    binding = SimpleNamespace(diffsynth_model_paths=lambda: ["/models/wan.safetensors"])
    command = build_command(request, settings, tmp_path, tmp_path / "metadata.csv", binding, tmp_path / "output")
    joined = " ".join(command)
    assert "--lora_base_model dit" in joined
    assert "--remove_prefix_in_ckpt pipe.dit." in joined
    assert "cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2" in joined
    assert "--save_steps 400" in joined
    assert "--lora_base_model vace" not in joined
    assert "pipe.vace." not in joined
    assert DIT_TRAINING_PROFILE == "wan_dit_identity_video_v1"
    assert DIT_CHECKPOINT_STEPS == (400, 600, 800)
