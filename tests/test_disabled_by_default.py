import pytest
from identity_worker.config import Settings
from identity_worker.errors import WorkerError

def test_training_disabled_by_default():
    with pytest.raises(WorkerError): Settings().validate_runtime()
