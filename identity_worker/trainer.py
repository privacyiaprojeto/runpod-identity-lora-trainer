from __future__ import annotations
import json, subprocess
from pathlib import Path
from .errors import WorkerError

def build_command(request, settings, dataset_root: Path, metadata_path: Path, model_paths: list[str], output_dir: Path) -> list[str]:
    t=request.payload['training']
    return ['accelerate','launch',str(settings.diffsynth_root/'examples/wanvideo/model_training/train.py'),
      '--dataset_base_path',str(dataset_root),'--dataset_metadata_path',str(metadata_path),
      '--data_file_keys','video,vace_video,vace_reference_image','--height',str(t['height']),'--width',str(t['width']),
      '--num_frames',str(t['num_frames']),'--dataset_repeat',str(t['dataset_repeat']),'--model_paths',json.dumps(model_paths),
      '--learning_rate',str(t['learning_rate']),'--num_epochs',str(t['num_epochs']),'--remove_prefix_in_ckpt','pipe.vace.',
      '--output_path',str(output_dir),'--lora_base_model','vace','--lora_target_modules',','.join(t['target_modules']),
      '--lora_rank',str(t['lora_rank']),'--extra_inputs','vace_video,vace_reference_image','--use_gradient_checkpointing_offload']

def run_training(command: list[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True,exist_ok=True)
    try:
        subprocess.run(command,check=True)
    except subprocess.CalledProcessError as exc:
        raise WorkerError('DIFFSYNTH_TRAINING_FAILED', f'Treinamento encerrou com código {exc.returncode}.') from exc
    candidates=sorted(output_dir.rglob('*.safetensors'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not candidates:
        raise WorkerError('ADAPTER_NOT_FOUND','Treinamento terminou sem adapter .safetensors.')
    return candidates[0]
