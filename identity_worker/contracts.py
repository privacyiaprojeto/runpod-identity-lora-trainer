from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from .errors import WorkerError

UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.I)
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
CONTRACT_VERSION = 'privacy-identity-lora-training-v2'
PREVIEW_CONTRACT_VERSION = 'privacy-identity-lora-review-kit-v2'
PREVIEW_VIDEO_KEY = 'video_walk_turn_smile'
PREVIEW_IMAGE_KEYS = ('image_crying', 'image_sensual', 'image_lollipop')
DIT_TRAINING_PROFILE = 'wan_dit_identity_video_poc_v1'
DIT_TARGET_MODULES = ('cross_attn.q', 'cross_attn.k', 'cross_attn.v', 'cross_attn.o', 'ffn.0', 'ffn.2')
DIT_OPTIMIZER_STEPS = 800
DIT_CHECKPOINT_STEPS = (400, 600, 800)


def _text(value: Any) -> str:
    return str(value or '').strip()


def _private_ref(value: dict[str, Any]) -> bool:
    bucket = _text(value.get('bucket'))
    key = _text(value.get('key'))
    return bool(bucket and key and not bucket.startswith(('http://','https://')) and not key.startswith(('http://','https://','/')))


def _parse_expiry(value: Any) -> datetime | None:
    raw = _text(value)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


@dataclass(frozen=True)
class TrainingRequest:
    payload: dict[str, Any]
    request_id: str
    actor_profile_id: str
    training_run_id: str
    output_bucket: str
    output_prefix: str
    smoke_expires_at: str


def parse_training_request(event: dict[str, Any]) -> TrainingRequest:
    payload = event.get('input') if isinstance(event.get('input'), dict) else event
    if payload.get('contract_version') != CONTRACT_VERSION:
        raise WorkerError('UNSUPPORTED_CONTRACT', 'Contrato de treinamento incompatível.')
    if payload.get('execution_mode') != 'controlled_training_smoke':
        raise WorkerError('INVALID_EXECUTION_MODE', 'Modo de execução inválido.')
    if not _text(payload.get('request_id')):
        raise WorkerError('REQUEST_ID_REQUIRED', 'Identificador da solicitação ausente.')
    actor_id = _text(payload.get('actor_profile_id'))
    run_id = _text(payload.get('training_run_id'))
    if not UUID_RE.match(actor_id) or not UUID_RE.match(run_id):
        raise WorkerError('INVALID_SCOPE', 'Ator ou run inválido.')
    if not SHA_RE.match(_text(payload.get('dataset_manifest_sha256')).lower()):
        raise WorkerError('INVALID_DATASET_SIGNATURE', 'Assinatura do conjunto inválida.')
    samples = (payload.get('dataset') or {}).get('samples') or []
    if len(samples) < 15:
        raise WorkerError('INSUFFICIENT_SAMPLES', 'O contrato exige ao menos 15 amostras internas.')
    for sample in samples:
        if not _private_ref(sample.get('video_source') or {}) or not _private_ref(sample.get('reference_image_source') or {}):
            raise WorkerError('PUBLIC_REFERENCE_FORBIDDEN', 'Somente referências privadas são aceitas.')
        if not SHA_RE.match(_text(sample.get('video_sha256')).lower()) or not SHA_RE.match(_text(sample.get('reference_image_sha256')).lower()):
            raise WorkerError('INVALID_ASSET_CHECKSUM', 'Checksum de material inválido.')
    model = payload.get('model') or {}
    if not SHA_RE.match(_text(model.get('fingerprint_sha256')).lower()) or len(model.get('artifacts') or []) != 9:
        raise WorkerError('INVALID_MODEL_LOCK', 'Lock do modelo-base inválido.')

    training = payload.get('training') or {}
    exact_training = {
        'profile': DIT_TRAINING_PROFILE,
        'width': 832,
        'height': 480,
        'num_frames': 17,
        'optimizer_steps': DIT_OPTIMIZER_STEPS,
        'num_epochs': 1,
        'learning_rate': 0.00005,
        'lora_rank': 32,
        'lora_alpha': 32,
        'lora_base_model': 'dit',
        'remove_prefix_in_ckpt': 'pipe.dit.',
        'vace_frozen': True,
        'automatic_retry': False,
    }
    mismatched = [key for key, expected in exact_training.items() if training.get(key) != expected]
    if mismatched:
        raise WorkerError('INVALID_DIT_TRAINING_PROFILE', f"O contrato não corresponde ao POC DiT controlado: {','.join(mismatched)}.")
    if int(training.get('dataset_repeat') or 0) < 54:
        raise WorkerError('INSUFFICIENT_TRAINING_ITERATIONS', 'O dataset_repeat não alcança a janela de 800 passos.')
    if tuple(training.get('checkpoint_steps') or ()) != DIT_CHECKPOINT_STEPS:
        raise WorkerError('INVALID_CHECKPOINT_PLAN', 'Os checkpoints precisam ser exatamente 400, 600 e 800.')
    if tuple(training.get('target_modules') or ()) != DIT_TARGET_MODULES:
        raise WorkerError('INVALID_DIT_TARGET_MODULES', 'Os módulos LoRA não correspondem ao gerador principal Wan DiT.')
    if any('vace' in _text(item).lower() or 'self_attn' in _text(item).lower() for item in training.get('target_modules') or []):
        raise WorkerError('FORBIDDEN_TRAINING_TARGET', 'O ramo VACE e a self-attention devem permanecer congelados.')

    safety = payload.get('safety') or {}
    required_safety = {
        'actor_scoped': True,
        'private_storage_only': True,
        'public_urls_forbidden': True,
        'product_release_allowed': False,
        'inference_injection_allowed': False,
        'automatic_retry_allowed': False,
        'one_shot_smoke': True,
    }
    if any(safety.get(key) is not expected for key, expected in required_safety.items()):
        raise WorkerError('INVALID_SAFETY_CONTRACT', 'Contrato de segurança incompleto ou incompatível.')
    smoke = payload.get('smoke') or {}
    expiry = _parse_expiry(smoke.get('expires_at'))
    if smoke.get('enabled') is not True or smoke.get('one_shot') is not True or int(smoke.get('max_jobs') or 0) != 1:
        raise WorkerError('INVALID_SMOKE_CONTRACT', 'A execução real precisa ser one-shot.')
    if _text(smoke.get('actor_profile_id')) != actor_id or _text(smoke.get('training_run_id')) != run_id:
        raise WorkerError('SMOKE_SCOPE_MISMATCH', 'O escopo do smoke não corresponde ao ator e ao run.')
    if not expiry or expiry <= datetime.now(timezone.utc):
        raise WorkerError('SMOKE_WINDOW_EXPIRED', 'A janela controlada do smoke expirou.')
    output = payload.get('output') or {}
    if output.get('public') is not False or not _text(output.get('bucket')) or not _text(output.get('prefix')):
        raise WorkerError('PRIVATE_OUTPUT_REQUIRED', 'Destino privado obrigatório.')
    expected_scope = f'/{actor_id}/{run_id}'
    if expected_scope not in f"/{_text(output.get('prefix')).strip('/')}":
        raise WorkerError('OUTPUT_SCOPE_MISMATCH', 'Destino do adapter não está isolado pelo ator e pelo run.')
    return TrainingRequest(payload, _text(payload.get('request_id')), actor_id, run_id, _text(output.get('bucket')), _text(output.get('prefix')), expiry.isoformat())


@dataclass(frozen=True)
class PreviewRequest:
    payload: dict[str, Any]
    request_id: str
    actor_profile_id: str
    training_run_id: str
    adapter_id: str
    output_bucket: str
    output_prefix: str
    smoke_expires_at: str


def parse_preview_request(event: dict[str, Any]) -> PreviewRequest:
    payload = event.get('input') if isinstance(event.get('input'), dict) else event
    if payload.get('contract_version') != PREVIEW_CONTRACT_VERSION:
        raise WorkerError('UNSUPPORTED_PREVIEW_CONTRACT', 'Contrato do kit de validação incompatível.')
    if payload.get('execution_mode') != 'controlled_identity_qa_kit':
        raise WorkerError('INVALID_PREVIEW_MODE', 'Modo do kit de validação inválido.')
    request_id = _text(payload.get('request_id'))
    actor_id = _text(payload.get('actor_profile_id'))
    run_id = _text(payload.get('training_run_id'))
    adapter_id = _text(payload.get('adapter_id'))
    if not request_id:
        raise WorkerError('PREVIEW_REQUEST_ID_REQUIRED', 'Identificador do kit de validação ausente.')
    if not all(UUID_RE.match(value) for value in (actor_id, run_id, adapter_id)):
        raise WorkerError('INVALID_PREVIEW_SCOPE', 'Ator, run ou adapter inválido.')

    adapter = payload.get('adapter') or {}
    if not _private_ref(adapter) or not SHA_RE.match(_text(adapter.get('sha256')).lower()) or int(adapter.get('byte_size') or 0) <= 0:
        raise WorkerError('INVALID_PRIVATE_ADAPTER', 'O adapter privado do kit é inválido.')
    source = payload.get('source') or {}
    for name in ('control_video', 'reference_image'):
        item = source.get(name) or {}
        if not _private_ref(item) or not SHA_RE.match(_text(item.get('sha256')).lower()):
            raise WorkerError('INVALID_PREVIEW_SOURCE', 'O kit exige vídeo e foto privados com checksum.')

    model = payload.get('model') or {}
    if not SHA_RE.match(_text(model.get('fingerprint_sha256')).lower()) or len(model.get('artifacts') or []) != 9:
        raise WorkerError('INVALID_PREVIEW_MODEL_LOCK', 'Lock do modelo-base inválido para o kit.')

    preview = payload.get('preview') or {}
    if preview.get('profile') != 'identity_private_qa_kit_v2' or preview.get('one_qa_kit') is not True or int(preview.get('asset_count') or 0) != 4:
        raise WorkerError('INVALID_PREVIEW_PROFILE', 'O kit deve usar o perfil universal homologado com quatro evidências.')
    video = preview.get('video') or {}
    video_required = {'asset_key': PREVIEW_VIDEO_KEY, 'width': 1024, 'height': 576, 'num_frames': 61, 'fps': 12}
    if any(video.get(key) != expected for key, expected in video_required.items()):
        raise WorkerError('INVALID_PREVIEW_VIDEO_PROFILE', 'O vídeo QA deve usar 1024x576, 61 quadros e 12 fps.')
    if int(video.get('num_inference_steps') or 0) < 20 or int(video.get('num_inference_steps') or 0) > 30:
        raise WorkerError('INVALID_PREVIEW_STEPS', 'Passos do vídeo QA fora do limite seguro.')

    images = preview.get('images') or []
    if not isinstance(images, list) or len(images) != 3 or tuple(item.get('asset_key') for item in images) != PREVIEW_IMAGE_KEYS:
        raise WorkerError('INVALID_PREVIEW_IMAGE_SET', 'O kit deve conter exatamente as três imagens QA homologadas.')
    for item in images:
        if item.get('width') != 768 or item.get('height') != 1024 or item.get('num_frames') != 1:
            raise WorkerError('INVALID_PREVIEW_IMAGE_PROFILE', 'Cada imagem QA deve usar 768x1024 e um quadro.')
        if int(item.get('num_inference_steps') or 0) < 20 or int(item.get('num_inference_steps') or 0) > 32:
            raise WorkerError('INVALID_PREVIEW_IMAGE_STEPS', 'Passos da imagem QA fora do limite seguro.')
    if not (0.1 <= float(preview.get('lora_strength') or 0) <= 1.2):
        raise WorkerError('INVALID_PREVIEW_STRENGTH', 'Força do adapter fora do limite seguro.')

    safety = payload.get('safety') or {}
    required_safety = {
        'actor_scoped': True,
        'run_scoped': True,
        'adapter_scoped': True,
        'private_storage_only': True,
        'public_urls_forbidden': True,
        'product_release_allowed': False,
        'automatic_retry_allowed': False,
        'one_shot_smoke': True,
        'approval_allowed': False,
        'qa_kit_only': True,
    }
    if any(safety.get(key) is not expected for key, expected in required_safety.items()):
        raise WorkerError('INVALID_PREVIEW_SAFETY', 'Contrato de segurança do kit incompleto.')

    smoke = payload.get('smoke') or {}
    expiry = _parse_expiry(smoke.get('expires_at'))
    if smoke.get('enabled') is not True or smoke.get('one_shot') is not True or int(smoke.get('max_jobs') or 0) != 1:
        raise WorkerError('INVALID_PREVIEW_SMOKE', 'O kit real precisa ser one-shot.')
    if _text(smoke.get('actor_profile_id')) != actor_id or _text(smoke.get('training_run_id')) != run_id or _text(smoke.get('adapter_id')) != adapter_id:
        raise WorkerError('PREVIEW_SCOPE_MISMATCH', 'O escopo do kit não corresponde à autorização.')
    if not expiry or expiry <= datetime.now(timezone.utc):
        raise WorkerError('PREVIEW_WINDOW_EXPIRED', 'A janela controlada do kit expirou.')

    output = payload.get('output') or {}
    if output.get('public') is not False or not _text(output.get('bucket')) or not _text(output.get('prefix')):
        raise WorkerError('PRIVATE_PREVIEW_OUTPUT_REQUIRED', 'Destino privado obrigatório para o kit.')
    expected_scope = f'/{actor_id}/{run_id}/{adapter_id}'
    if expected_scope not in f"/{_text(output.get('prefix')).strip('/')}":
        raise WorkerError('PREVIEW_OUTPUT_SCOPE_MISMATCH', 'Destino do kit não está isolado por ator, run e adapter.')
    return PreviewRequest(payload, request_id, actor_id, run_id, adapter_id, _text(output.get('bucket')), _text(output.get('prefix')), expiry.isoformat())
