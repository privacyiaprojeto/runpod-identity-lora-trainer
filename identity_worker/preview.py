from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .errors import WorkerError
from .storage import download_private


def materialize_preview_inputs(request, work_dir: Path, s3) -> dict[str, Path]:
    adapter = request.payload['adapter']
    source = request.payload['source']
    root = work_dir / 'preview-inputs'
    return {
        'adapter': download_private(s3, adapter['bucket'], adapter['key'], root / 'identity_adapter.safetensors', adapter['sha256']),
        'control_video': download_private(s3, source['control_video']['bucket'], source['control_video']['key'], root / 'control_source.mp4', source['control_video']['sha256']),
        'reference_image': download_private(s3, source['reference_image']['bucket'], source['reference_image']['key'], root / 'reference_source.jpg', source['reference_image']['sha256']),
    }


def _run(command: list[str], code: str, message: str) -> None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except Exception as exc:
        raise WorkerError(code, message, retryable=False) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or '').strip()[-800:]
        raise WorkerError(code, f'{message} {detail}'.strip(), retryable=False)


def _prepare_control_video(source: Path, destination: Path, width: int, height: int, fps: int, frames: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
        '-vf', f'fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}',
        '-frames:v', str(frames), '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
        str(destination),
    ], 'PREVIEW_CONTROL_VIDEO_FAILED', 'Não foi possível preparar o vídeo neutro da prévia.')
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise WorkerError('PREVIEW_CONTROL_VIDEO_EMPTY', 'O vídeo de controle da prévia ficou vazio.')
    return destination


def _prepare_reference_image(source: Path, destination: Path, width: int, height: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-frames:v', '1', str(destination),
    ], 'PREVIEW_REFERENCE_IMAGE_FAILED', 'Não foi possível preparar a foto de referência da prévia.')
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise WorkerError('PREVIEW_REFERENCE_IMAGE_EMPTY', 'A foto de referência da prévia ficou vazia.')
    return destination


def _local_model_config(ModelConfig, value: Any, **kwargs):
    attempts = (
        lambda: ModelConfig(value, **kwargs),
        lambda: ModelConfig(path=value, **kwargs),
        lambda: ModelConfig(model_path=value, **kwargs),
    )
    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last_error = exc
    raise WorkerError('PREVIEW_MODEL_CONFIG_FAILED', 'O DiffSynth não aceitou o binding local congelado da prévia.') from last_error


def run_preview(request, settings, model_binding, inputs: dict[str, Path], work_dir: Path) -> Path:
    preview = request.payload['preview']
    width = int(preview['width'])
    height = int(preview['height'])
    fps = int(preview['fps'])
    frames = int(preview['num_frames'])
    control_path = _prepare_control_video(inputs['control_video'], work_dir / 'prepared' / 'control.mp4', width, height, fps, frames)
    reference_path = _prepare_reference_image(inputs['reference_image'], work_dir / 'prepared' / 'reference.jpg', width, height)
    output_path = work_dir / 'output' / 'identity-review-preview.mp4'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    try:
        import torch
        from PIL import Image
        from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
        from diffsynth.utils.data import VideoData, save_video
    except Exception as exc:
        raise WorkerError('PREVIEW_RUNTIME_IMPORT_FAILED', 'O runtime de inferência da prévia não está completo.', retryable=False) from exc

    vram_config = {
        'offload_dtype': 'disk',
        'offload_device': 'disk',
        'onload_dtype': torch.bfloat16,
        'onload_device': 'cpu',
        'preparing_dtype': torch.bfloat16,
        'preparing_device': 'cuda',
        'computation_dtype': torch.bfloat16,
        'computation_device': 'cuda',
    }
    try:
        model_configs = [
            _local_model_config(ModelConfig, list(model_binding.diffusion_shards), **vram_config),
            _local_model_config(ModelConfig, model_binding.text_encoder_path, **vram_config),
            _local_model_config(ModelConfig, model_binding.vae_path, **vram_config),
        ]
        tokenizer_config = ModelConfig(
            model_id=model_binding.repository,
            origin_file_pattern='google/umt5-xxl/',
            revision=model_binding.revision,
        )
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device='cuda',
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            vram_limit=max(1.0, torch.cuda.mem_get_info('cuda')[1] / (1024 ** 3) - 2.0),
        )
        lora_target = getattr(pipe, 'vace', None) or getattr(pipe, 'dit', None)
        if lora_target is None:
            raise WorkerError('PREVIEW_LORA_TARGET_MISSING', 'O pipeline não expôs o componente VACE para aplicar a identidade.')
        pipe.load_lora(lora_target, str(inputs['adapter']), alpha=float(preview['lora_strength']))
        control = VideoData(str(control_path), height=height, width=width)
        reference = Image.open(reference_path).convert('RGB')
        generated = pipe(
            prompt=str(preview['prompt']),
            negative_prompt=str(preview.get('negative_prompt') or ''),
            vace_video=control,
            vace_reference_image=reference,
            vace_scale=float(preview.get('vace_scale') or 1.0),
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=int(preview['num_inference_steps']),
            seed=int(preview['seed']),
            tiled=True,
        )
        save_video(generated, str(output_path), fps=fps, quality=5)
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError('PREVIEW_INFERENCE_FAILED', 'A inferência curta da identidade falhou antes de produzir a prévia.', retryable=False) from exc

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise WorkerError('PREVIEW_OUTPUT_EMPTY', 'A prévia terminou sem produzir um vídeo válido.')
    return output_path
