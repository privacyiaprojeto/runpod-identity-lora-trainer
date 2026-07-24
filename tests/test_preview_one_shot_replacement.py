import json
from pathlib import Path
import pytest
from identity_worker.errors import WorkerError
from identity_worker.one_shot import reserve_preview_one_shot, update_one_shot

ACTOR_ID='11111111-1111-4111-8111-111111111111'
RUN_ID='22222222-2222-4222-8222-222222222222'
ADAPTER_ID='33333333-3333-4333-8333-333333333333'


def test_completed_short_preview_allows_exactly_one_explicit_qa_replacement(tmp_path: Path):
    first=reserve_preview_one_shot(tmp_path,ACTOR_ID,RUN_ID,ADAPTER_ID,'request-1')
    update_one_shot(first,'completed',previewDurationSeconds=1.133)
    second=reserve_preview_one_shot(tmp_path,ACTOR_ID,RUN_ID,ADAPTER_ID,'request-2',replacement_enabled=True,replacement_required_reason='PREVIEW_DURATION_TOO_SHORT',replacement_max_attempts=1)
    payload=json.loads(second.read_text())
    assert payload['replacementCount']==1
    assert payload['replacementOfRequestId']=='request-1'
    assert payload['automaticRetry'] is False
    assert len(list(tmp_path.glob(f'{ADAPTER_ID}.completed-replacement-1-*.json')))==1
    update_one_shot(second,'completed')
    with pytest.raises(WorkerError):
        reserve_preview_one_shot(tmp_path,ACTOR_ID,RUN_ID,ADAPTER_ID,'request-3',replacement_enabled=True,replacement_required_reason='PREVIEW_DURATION_TOO_SHORT',replacement_max_attempts=1)


def test_replacement_and_recovery_cannot_be_opened_together(tmp_path: Path):
    with pytest.raises(WorkerError) as error:
        reserve_preview_one_shot(tmp_path,ACTOR_ID,RUN_ID,ADAPTER_ID,'request-1',recovery_enabled=True,recovery_max_attempts=1,replacement_enabled=True,replacement_max_attempts=1)
    assert error.value.code == 'PREVIEW_RECOVERY_REPLACEMENT_CONFLICT'
