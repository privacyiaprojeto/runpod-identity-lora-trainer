import pytest
from identity_worker.config import Settings
from identity_worker.errors import WorkerError


def test_preview_disabled_by_default():
    with pytest.raises(WorkerError) as error:
        Settings().validate_preview_runtime()
    assert error.value.code == 'PREVIEW_DISABLED'
