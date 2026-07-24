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
    payload = {'schemaVersion':'privacy-identity-lora-smoke-lock-v1','actorProfileId':actor_profile_id,'trainingRunId':training_run_id,'requestId':request_id,'status':'reserved','reservedAt':_now()}
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
        raise WorkerError('PREVIEW_LOCK_INVALID', 'O lock anterior do kit não pôde ser validado.') from exc
    return payload if isinstance(payload, dict) else {}


def reserve_preview_one_shot(
    lock_root: Path, actor_profile_id: str, training_run_id: str, adapter_id: str, request_id: str, *,
    recovery_enabled: bool = False, recovery_required_error_code: str = 'PREVIEW_INFERENCE_FAILED', recovery_max_attempts: int = 0,
    replacement_enabled: bool = False, replacement_required_reason: str = 'PREVIEW_DURATION_TOO_SHORT', replacement_max_attempts: int = 0,
) -> Path:
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f'{adapter_id}.json'
    recovery_count = 0
    replacement_count = 0
    recovery_of_request_id = None
    recovery_of_error_code = None
    replacement_of_request_id = None
    archived_lock_path = None

    if recovery_enabled and replacement_enabled:
        raise WorkerError('PREVIEW_RECOVERY_REPLACEMENT_CONFLICT', 'Recuperação e substituição não podem ser usadas juntas.')

    if lock_path.exists():
        previous = _read_lock(lock_path)
        recovery_count = int(previous.get('recoveryCount') or 0)
        replacement_count = int(previous.get('replacementCount') or 0)
        same_scope = (
            previous.get('schemaVersion') == 'privacy-identity-lora-preview-lock-v1'
            and previous.get('actorProfileId') == actor_profile_id
            and previous.get('trainingRunId') == training_run_id
            and previous.get('adapterId') == adapter_id
        )
        recovery_allowed = (
            recovery_enabled is True and recovery_max_attempts == 1 and same_scope
            and previous.get('status') == 'failed' and previous.get('errorCode') == recovery_required_error_code
            and previous.get('retryable') is False and recovery_count < recovery_max_attempts
        )
        replacement_allowed = (
            replacement_enabled is True and replacement_max_attempts == 1 and same_scope
            and replacement_required_reason == 'PREVIEW_DURATION_TOO_SHORT'
            and previous.get('status') == 'completed' and replacement_count < replacement_max_attempts
        )
        if not recovery_allowed and not replacement_allowed:
            raise WorkerError('PREVIEW_ALREADY_CONSUMED', 'Este adapter já consumiu a execução autorizada do kit QA.')
        suffix = str(previous.get('requestId') or 'unknown').replace('/', '_')[:48]
        if recovery_allowed:
            recovery_of_request_id = previous.get('requestId')
            recovery_of_error_code = previous.get('errorCode')
            archived_lock_path = lock_root / f'{adapter_id}.failed-{recovery_count + 1}-{suffix}.json'
            recovery_count += 1
        else:
            replacement_of_request_id = previous.get('requestId')
            archived_lock_path = lock_root / f'{adapter_id}.completed-replacement-{replacement_count + 1}-{suffix}.json'
            replacement_count += 1
        try:
            os.replace(lock_path, archived_lock_path)
        except FileNotFoundError as exc:
            raise WorkerError('PREVIEW_RECOVERY_RACE_BLOCKED', 'Outra tentativa já tratou o lock anterior do kit.') from exc

    payload = {
        'schemaVersion': 'privacy-identity-lora-preview-lock-v1',
        'actorProfileId': actor_profile_id, 'trainingRunId': training_run_id, 'adapterId': adapter_id,
        'requestId': request_id, 'status': 'reserved', 'reservedAt': _now(),
        'recoveryCount': recovery_count, 'recoveryOfRequestId': recovery_of_request_id, 'recoveryOfErrorCode': recovery_of_error_code,
        'replacementCount': replacement_count, 'replacementOfRequestId': replacement_of_request_id,
        'replacementReason': replacement_required_reason if replacement_of_request_id else None,
        'automaticRetry': False,
    }
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise WorkerError('PREVIEW_ALREADY_CONSUMED', 'Este adapter já consumiu a execução autorizada do kit QA.') from exc
    except Exception:
        if archived_lock_path is not None and archived_lock_path.exists() and not lock_path.exists():
            os.replace(archived_lock_path, lock_path)
        raise
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return lock_path
