from identity_worker.trainer import _classify_failure


def test_missing_librosa_has_specific_retryable_code():
    error = _classify_failure("ModuleNotFoundError: No module named 'librosa'", 1)
    assert error.code == "TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING"
    assert error.retryable is True


def test_missing_soundfile_has_specific_retryable_code():
    error = _classify_failure("ModuleNotFoundError: No module named 'soundfile'", 1)
    assert error.code == "TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING"
    assert error.retryable is True


def test_import_error_is_retryable_runtime_failure():
    error = _classify_failure("ImportError: cannot import name 'is_offline_mode'", 1)
    assert error.code == "TRAINING_RUNTIME_IMPORT_FAILED"
    assert error.retryable is True


def test_cuda_oom_is_retryable():
    error = _classify_failure("CUDA out of memory", 1)
    assert error.code == "TRAINING_GPU_OUT_OF_MEMORY"
    assert error.retryable is True


def test_unknown_diffsynth_failure_is_retryable_but_not_hidden():
    error = _classify_failure("unexpected failure", 17)
    assert error.code == "DIFFSYNTH_TRAINING_FAILED"
    assert "17" in str(error)
    assert error.retryable is True
