import pytest

from identity_worker.errors import WorkerError
from identity_worker.preview import TOKENIZER_ORIGIN_FILE_PATTERN, build_tokenizer_config


class StrictModelConfig:
    def __init__(self, *, model_id, origin_file_pattern):
        self.model_id = model_id
        self.origin_file_pattern = origin_file_pattern


def test_tokenizer_config_uses_only_supported_model_config_arguments():
    config = build_tokenizer_config(StrictModelConfig, 'Wan-AI/Wan2.1-VACE-14B')
    assert config.model_id == 'Wan-AI/Wan2.1-VACE-14B'
    assert config.origin_file_pattern == TOKENIZER_ORIGIN_FILE_PATTERN


def test_tokenizer_config_rejects_missing_repository():
    with pytest.raises(WorkerError) as captured:
        build_tokenizer_config(StrictModelConfig, '')
    assert captured.value.code == 'PREVIEW_TOKENIZER_MODEL_ID_MISSING'


def test_tokenizer_config_classifies_signature_mismatch():
    class OldOrUnexpectedModelConfig:
        def __init__(self, *, model_id):
            self.model_id = model_id

    with pytest.raises(WorkerError) as captured:
        build_tokenizer_config(OldOrUnexpectedModelConfig, 'Wan-AI/Wan2.1-VACE-14B')
    assert captured.value.code == 'PREVIEW_TOKENIZER_CONFIG_INCOMPATIBLE'
