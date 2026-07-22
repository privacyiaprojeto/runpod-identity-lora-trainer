import json
from pathlib import Path
import pytest
from identity_worker.errors import WorkerError
from identity_worker.one_shot import reserve_one_shot, update_one_shot


def test_one_shot_lock_blocks_duplicate(tmp_path: Path):
    lock = reserve_one_shot(tmp_path, 'actor', 'run', 'request')
    assert lock.exists()
    with pytest.raises(WorkerError) as error:
        reserve_one_shot(tmp_path, 'actor', 'run', 'request-2')
    assert error.value.code == 'SMOKE_ALREADY_CONSUMED'


def test_one_shot_lock_records_terminal_state(tmp_path: Path):
    lock = reserve_one_shot(tmp_path, 'actor', 'run', 'request')
    update_one_shot(lock, 'completed', adapterSha256='a' * 64)
    payload = json.loads(lock.read_text(encoding='utf-8'))
    assert payload['status'] == 'completed'
    assert payload['adapterSha256'] == 'a' * 64
