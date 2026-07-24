from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from .errors import WorkerError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reserve_one_shot(lock_root: Path, actor_profile_id: str, training_run_id: str, request_id: str) -> Path:
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f'{training_run_id}.json'
    payload = {
        'schemaVersion': 'privacy-identity-lora-smoke-lock-v1',
        'actorProfileId': actor_profile_id,
        'trainingRunId': training_run_id,
        'requestId': request_id,
        'status': 'reserved',
        'reservedAt': _now(),
    }
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise WorkerError('SMOKE_ALREADY_CONSUMED', 'Este run já consumiu a única execução real autorizada.') from exc
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return lock_path


def update_one_shot(lock_path: Path, status: str, **fields) -> None:
    try:
        payload = json.loads(lock_path.read_text(encoding='utf-8'))
    except Exception:
        payload = {'schemaVersion': 'privacy-identity-lora-smoke-lock-v1'}
    payload.update(fields)
    payload['status'] = status
    payload['updatedAt'] = _now()
    temporary = lock_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(lock_path)


def _read_lock(lock_path: Path) -> dict:
    try:
        payload = json.loads(lock_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise WorkerError('PREVIEW_LOCK_INVALID', 'O lock anterior da prévia não pôde ser validado.') from exc
    return payload if isinstance(payload, dict) else {}


def reserve_preview_one_shot(
    lock_root: Path,
    actor_profile_id: str,
    training_run_id: str,
    adapter_id: str,
    request_id: str,
    *,
    recovery_enabled: bool = False,
    recovery_required_error_code: str = 'PREVIEW_INFERENCE_FAILED',
    recovery_max_attempts: int = 0,
) -> Path:
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f'{adapter_id}.json'
    recovery_count = 0
    recovery_of_request_id = None
    recovery_of_error_code = None
    archived_lock_path = None

    if lock_path.exists():
        previous = _read_lock(lock_path)
        recovery_count = int(previous.get('recoveryCount') or 0)
        allowed = (
            recovery_enabled is True
            and recovery_max_attempts == 1
            and previous.get('schemaVersion') == 'privacy-identity-lora-preview-lock-v1'
            and previous.get('actorProfileId') == actor_profile_id
            and previous.get('trainingRunId') == training_run_id
            and previous.get('adapterId') == adapter_id
            and previous.get('status') == 'failed'
            and previous.get('errorCode') == recovery_required_error_code
            and previous.get('retryable') is False
            and recovery_count < recovery_max_attempts
        )
        if not allowed:
            raise WorkerError('PREVIEW_ALREADY_CONSUMED', 'Este adapter já consumiu a única prévia autorizada.')
        recovery_of_request_id = previous.get('requestId')
        recovery_of_error_code = previous.get('errorCode')
        archive_suffix = str(recovery_of_request_id or 'unknown').replace('/', '_')[:48]
        archived_lock_path = lock_root / f'{adapter_id}.failed-{recovery_count + 1}-{archive_suffix}.json'
        try:
            os.replace(lock_path, archived_lock_path)
        except FileNotFoundError as exc:
            raise WorkerError('PREVIEW_RECOVERY_RACE_BLOCKED', 'Outra tentativa já tratou o lock com falha da prévia.') from exc
        recovery_count += 1

    payload = {
        'schemaVersion': 'privacy-identity-lora-preview-lock-v1',
        'actorProfileId': actor_profile_id,
        'trainingRunId': training_run_id,
        'adapterId': adapter_id,
        'requestId': request_id,
        'status': 'reserved',
        'reservedAt': _now(),
        'recoveryCount': recovery_count,
        'recoveryOfRequestId': recovery_of_request_id,
        'recoveryOfErrorCode': recovery_of_error_code,
        'automaticRetry': False,
    }
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise WorkerError('PREVIEW_ALREADY_CONSUMED', 'Este adapter já consumiu a única prévia autorizada.') from exc
    except Exception:
        if archived_lock_path is not None and archived_lock_path.exists() and not lock_path.exists():
            os.replace(archived_lock_path, lock_path)
        raise
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return lock_path
