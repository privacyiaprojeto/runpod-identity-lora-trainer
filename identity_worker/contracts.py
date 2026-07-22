from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from .errors import WorkerError

UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.I)
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
CONTRACT_VERSION = 'privacy-identity-lora-training-v2'


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
