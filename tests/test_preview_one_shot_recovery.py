import json
from pathlib import Path

import pytest

from identity_worker.errors import WorkerError
from identity_worker.one_shot import reserve_preview_one_shot, update_one_shot

ACTOR_ID = '11111111-1111-4111-8111-111111111111'
RUN_ID = '22222222-2222-4222-8222-222222222222'
ADAPTER_ID = '33333333-3333-4333-8333-333333333333'


def test_failed_preview_allows_exactly_one_explicit_recovery(tmp_path: Path):
    first = reserve_preview_one_shot(tmp_path, ACTOR_ID, RUN_ID, ADAPTER_ID, 'request-1')
    update_one_shot(first, 'failed', errorCode='PREVIEW_INFERENCE_FAILED', retryable=False)

    second = reserve_preview_one_shot(
        tmp_path, ACTOR_ID, RUN_ID, ADAPTER_ID, 'request-2',
        recovery_enabled=True,
        recovery_required_error_code='PREVIEW_INFERENCE_FAILED',
        recovery_max_attempts=1,
    )
    payload = json.loads(second.read_text(encoding='utf-8'))
    assert payload['recoveryCount'] == 1
    assert payload['recoveryOfRequestId'] == 'request-1'
    assert payload['automaticRetry'] is False
    assert len(list(tmp_path.glob(f'{ADAPTER_ID}.failed-1-*.json'))) == 1

    update_one_shot(second, 'failed', errorCode='PREVIEW_INFERENCE_FAILED', retryable=False)
    with pytest.raises(WorkerError) as captured:
        reserve_preview_one_shot(
            tmp_path, ACTOR_ID, RUN_ID, ADAPTER_ID, 'request-3',
            recovery_enabled=True,
            recovery_required_error_code='PREVIEW_INFERENCE_FAILED',
            recovery_max_attempts=1,
        )
    assert captured.value.code == 'PREVIEW_ALREADY_CONSUMED'


def test_recovery_rejects_different_error_or_scope(tmp_path: Path):
    first = reserve_preview_one_shot(tmp_path, ACTOR_ID, RUN_ID, ADAPTER_ID, 'request-1')
    update_one_shot(first, 'failed', errorCode='OTHER_FAILURE', retryable=False)
    with pytest.raises(WorkerError):
        reserve_preview_one_shot(
            tmp_path, ACTOR_ID, RUN_ID, ADAPTER_ID, 'request-2',
            recovery_enabled=True, recovery_required_error_code='PREVIEW_INFERENCE_FAILED', recovery_max_attempts=1,
        )
