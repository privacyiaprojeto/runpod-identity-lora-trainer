from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .errors import WorkerError
from .storage import download_private

TOKENIZER_ORIGIN_FILE_PATTERN = 'google/umt5-xxl/'


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
        '-frames:v', str(frames), '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17',
        str(destination),
    ], 'PREVIEW_CONTROL_VIDEO_FAILED', 'Não foi possível preparar o vídeo de controle do kit QA.')
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise WorkerError('PREVIEW_CONTROL_VIDEO_EMPTY', 'O vídeo de controle do kit ficou vazio.')
    return destination


def _prepare_reference_image(source: Path, destination: Path, width: int, height: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-frames:v', '1', str(destination),
    ], 'PREVIEW_REFERENCE_IMAGE_FAILED', 'Não foi possível preparar a foto de referência do kit QA.')
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise WorkerError('PREVIEW_REFERENCE_IMAGE_EMPTY', 'A foto de referência do kit ficou vazia.')
    return destination


def _prepare_static_control(reference: Path, destination: Path, width: int, height: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-loop', '1', '-i', str(reference),
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}',
        '-frames:v', '1', '-r', '1', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '15',
        str(destination),
    ], 'PREVIEW_STATIC_CONTROL_FAILED', 'Não foi possível preparar o controle estático da imagem QA.')
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise WorkerError('PREVIEW_STATIC_CONTROL_EMPTY', 'O controle estático da imagem QA ficou vazio.')
    return destination


def build_tokenizer_config(ModelConfig, repository: str):
    model_id = str(repository or '').strip()
    if not model_id:
        raise WorkerError('PREVIEW_TOKENIZER_MODEL_ID_MISSING', 'O modelo-base do kit não informou um repositório válido.', retryable=False)
    try:
        return ModelConfig(model_id=model_id, origin_file_pattern=TOKENIZER_ORIGIN_FILE_PATTERN)
    except TypeError as exc:
        raise WorkerError('PREVIEW_TOKENIZER_CONFIG_INCOMPATIBLE', 'O DiffSynth instalado não aceita o contrato homologado do tokenizer.', retryable=False) from exc


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
    raise WorkerError('PREVIEW_MODEL_CONFIG_FAILED', 'O DiffSynth não aceitou o binding local congelado do kit.') from last_error


def _save_generated_image(generated, destination: Path, Image) -> Path:
    frame = generated[0] if isinstance(generated, (list, tuple)) and generated else generated
    if isinstance(frame, Image.Image):
        image = frame.convert('RGB')
    elif hasattr(frame, 'cpu') and hasattr(frame, 'numpy'):
        array = frame.detach().cpu().numpy() if hasattr(frame, 'detach') else frame.cpu().numpy()
        image = Image.fromarray(array).convert('RGB')
    else:
        try:
            image = Image.fromarray(frame).convert('RGB')
        except Exception as exc:
            raise WorkerError('PREVIEW_IMAGE_OUTPUT_INVALID', 'O runtime não retornou uma imagem QA válida.', retryable=False) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format='PNG', optimize=True)
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise WorkerError('PREVIEW_IMAGE_OUTPUT_EMPTY', 'A imagem QA terminou vazia.', retryable=False)
    return destination


def run_qa_kit(request, settings, model_binding, inputs: dict[str, Path], work_dir: Path) -> dict[str, dict[str, Any]]:
    preview = request.payload['preview']
    output_root = work_dir / 'output'
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    try:
        import torch
        from PIL import Image
        from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
        from diffsynth.utils.data import VideoData, save_video
    except Exception as exc:
        raise WorkerError('PREVIEW_RUNTIME_IMPORT_FAILED', 'O runtime de inferência do kit QA não está completo.', retryable=False) from exc

    vram_config = {
        'offload_dtype': 'disk', 'offload_device': 'disk',
        'onload_dtype': torch.bfloat16, 'onload_device': 'cpu',
        'preparing_dtype': torch.bfloat16, 'preparing_device': 'cuda',
        'computation_dtype': torch.bfloat16, 'computation_device': 'cuda',
    }
    try:
        model_configs = [
            _local_model_config(ModelConfig, list(model_binding.diffusion_shards), **vram_config),
            _local_model_config(ModelConfig, model_binding.text_encoder_path, **vram_config),
            _local_model_config(ModelConfig, model_binding.vae_path, **vram_config),
        ]
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device='cuda',
            model_configs=model_configs,
            tokenizer_config=build_tokenizer_config(ModelConfig, model_binding.repository),
            vram_limit=max(1.0, torch.cuda.mem_get_info('cuda')[1] / (1024 ** 3) - 2.0),
        )
        lora_target = getattr(pipe, 'vace', None) or getattr(pipe, 'dit', None)
        if lora_target is None:
            raise WorkerError('PREVIEW_LORA_TARGET_MISSING', 'O pipeline não expôs o componente VACE para aplicar a identidade.')
        pipe.load_lora(lora_target, str(inputs['adapter']), alpha=float(preview['lora_strength']))

        video_profile = preview['video']
        video_control_path = _prepare_control_video(
            inputs['control_video'], work_dir / 'prepared' / 'video-control.mp4',
            int(video_profile['width']), int(video_profile['height']), int(video_profile['fps']), int(video_profile['num_frames']),
        )
        video_reference_path = _prepare_reference_image(
            inputs['reference_image'], work_dir / 'prepared' / 'video-reference.jpg',
            int(video_profile['width']), int(video_profile['height']),
        )
        generated_video = pipe(
            prompt=str(video_profile['prompt']), negative_prompt=str(video_profile.get('negative_prompt') or ''),
            vace_video=VideoData(str(video_control_path), height=int(video_profile['height']), width=int(video_profile['width'])),
            vace_reference_image=Image.open(video_reference_path).convert('RGB'),
            vace_scale=float(video_profile.get('vace_scale') or 1.0),
            height=int(video_profile['height']), width=int(video_profile['width']), num_frames=int(video_profile['num_frames']),
            num_inference_steps=int(video_profile['num_inference_steps']), seed=int(video_profile['seed']), tiled=True,
        )
        video_path = output_root / f"{video_profile['asset_key']}.mp4"
        save_video(generated_video, str(video_path), fps=int(video_profile['fps']), quality=7)
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise WorkerError('PREVIEW_VIDEO_OUTPUT_EMPTY', 'O vídeo QA terminou vazio.')

        assets: dict[str, dict[str, Any]] = {
            video_profile['asset_key']: {
                'path': video_path, 'kind': 'video', 'content_type': 'video/mp4',
                'width': int(video_profile['width']), 'height': int(video_profile['height']),
                'num_frames': int(video_profile['num_frames']), 'fps': int(video_profile['fps']),
                'duration_seconds': round(int(video_profile['num_frames']) / int(video_profile['fps']), 3),
                'label': str(video_profile['label']),
            }
        }

        for image_profile in preview['images']:
            width, height = int(image_profile['width']), int(image_profile['height'])
            reference_path = _prepare_reference_image(
                inputs['reference_image'], work_dir / 'prepared' / f"{image_profile['asset_key']}-reference.png", width, height,
            )
            static_control = _prepare_static_control(
                reference_path, work_dir / 'prepared' / f"{image_profile['asset_key']}-control.mp4", width, height,
            )
            generated_image = pipe(
                prompt=str(image_profile['prompt']), negative_prompt=str(image_profile.get('negative_prompt') or ''),
                vace_video=VideoData(str(static_control), height=height, width=width),
                vace_reference_image=Image.open(reference_path).convert('RGB'),
                vace_scale=float(image_profile.get('vace_scale') or 1.0),
                height=height, width=width, num_frames=1,
                num_inference_steps=int(image_profile['num_inference_steps']), seed=int(image_profile['seed']), tiled=True,
            )
            image_path = _save_generated_image(generated_image, output_root / f"{image_profile['asset_key']}.png", Image)
            assets[image_profile['asset_key']] = {
                'path': image_path, 'kind': 'image', 'content_type': 'image/png',
                'width': width, 'height': height, 'num_frames': 1, 'fps': None,
                'duration_seconds': None, 'label': str(image_profile['label']),
            }
        return assets
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError('PREVIEW_INFERENCE_FAILED', 'A inferência do kit QA falhou antes de produzir todas as evidências.', retryable=False) from exc
