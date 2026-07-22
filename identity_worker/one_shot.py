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
