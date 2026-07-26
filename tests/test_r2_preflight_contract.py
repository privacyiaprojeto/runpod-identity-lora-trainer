from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from identity_worker.contracts import R2_PREFLIGHT_CONTRACT_VERSION, parse_r2_preflight_request
from identity_worker.errors import WorkerError

ACTOR_ID = '11111111-1111-4111-8111-111111111111'
RUN_ID = '22222222-2222-4222-8222-222222222222'
BUCKET = 'privacy-media'


def event() -> dict:
    expires = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    base = f'vault/actor-mapping/actor-{ACTOR_ID}'
    return {
        'input': {
            'contract_version': R2_PREFLIGHT_CONTRACT_VERSION,
            'execution_mode': 'private_r2_metadata_preflight',
            'request_id': 'r2-preflight-test',
            'actor_profile_id': ACTOR_ID,
            'training_run_id': RUN_ID,
            'dataset_manifest_sha256': 'c' * 64,
            'bucket': BUCKET,
            'objects': [
                {
                    'sample_id': 'sample-01',
                    'role': 'video_source',
                    'source': {'bucket': BUCKET, 'key': f'{base}/video.mp4'},
                    'expected_sha256': 'a' * 64,
                },
                {
                    'sample_id': 'sample-01',
                    'role': 'reference_image_source',
                    'source': {'bucket': BUCKET, 'key': f'{base}/image.jpg'},
                    'expected_sha256': 'b' * 64,
                },
            ],
            'smoke': {
                'enabled': True,
                'one_shot': True,
                'max_jobs': 1,
                'actor_profile_id': ACTOR_ID,
                'training_run_id': RUN_ID,
                'expires_at': expires,
            },
            'safety': {
                'actor_scoped': True,
                'run_scoped': True,
                'private_storage_only': True,
                'metadata_only': True,
                'download_allowed': False,
                'write_allowed': False,
                'delete_allowed': False,
                'training_allowed': False,
                'model_load_allowed': False,
                'automatic_retry_allowed': False,
                'one_shot_smoke': True,
            },
        }
    }


def test_r2_preflight_contract_accepts_private_metadata_scope():
    request = parse_r2_preflight_request(event())
    assert request.actor_profile_id == ACTOR_ID
    assert request.training_run_id == RUN_ID
    assert request.bucket == BUCKET
    assert len(request.objects) == 2


def test_r2_preflight_contract_rejects_public_reference():
    payload = event()
    payload['input']['objects'][0]['source']['key'] = 'https://example.com/video.mp4'
    with pytest.raises(WorkerError) as error:
        parse_r2_preflight_request(payload)
    assert error.value.code == 'R2_PREFLIGHT_PUBLIC_REFERENCE_FORBIDDEN'


def test_r2_preflight_contract_rejects_actor_key_scope_mismatch():
    payload = event()
    payload['input']['objects'][0]['source']['key'] = 'vault/actor-mapping/actor-other/video.mp4'
    with pytest.raises(WorkerError) as error:
        parse_r2_preflight_request(payload)
    assert error.value.code == 'R2_PREFLIGHT_ACTOR_KEY_SCOPE_MISMATCH'


def test_r2_preflight_contract_rejects_write_permission():
    payload = event()
    payload['input']['safety']['write_allowed'] = True
    with pytest.raises(WorkerError) as error:
        parse_r2_preflight_request(payload)
    assert error.value.code == 'INVALID_R2_PREFLIGHT_SAFETY'
