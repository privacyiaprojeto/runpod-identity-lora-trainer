from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evaluate():
    requirements = (ROOT / 'requirements.txt').read_text(encoding='utf-8')
    dockerfile = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    handler = (ROOT / 'handler.py').read_text(encoding='utf-8')
    script = (ROOT / 'scripts' / 'poc' / 'TEST_RUNPOD_TRANSPORT_SMOKE.ps1').read_text(encoding='utf-8')

    checks = [
        ('runpod_sdk_pinned_1_11_0', 'runpod==1.11.0' in requirements and 'runpod==1.7.13' not in requirements),
        ('runpod_sdk_dependencies_compatible', 'boto3==1.43.51' in requirements and 'requests==2.34.2' in requirements and 'boto3==1.35.36' not in requirements and 'requests==2.32.3' not in requirements),
        ('docker_verifies_sdk_version', "RUNPOD_DEPENDENCY_BUILD_OK" in dockerfile and "'runpod':'1.11.0'" in dockerfile and "'boto3':'1.43.51'" in dockerfile and "'requests':'2.34.2'" in dockerfile),
        ('transport_contract_declared', "privacy-identity-lora-transport-smoke-v1" in handler),
        ('transport_disabled_by_default', "PRIVACY_LORA_TRANSPORT_SMOKE_ENABLED', 'false'" in handler),
        ('transport_requires_training_closed', 'TRANSPORT_SMOKE_REQUIRES_TRAINING_CLOSED' in handler),
        ('transport_requires_preview_closed', 'TRANSPORT_SMOKE_REQUIRES_PREVIEW_CLOSED' in handler),
        ('transport_branches_before_training', handler.index('contract_version == TRANSPORT_SMOKE_CONTRACT_VERSION') < handler.index('return _handle_training(event)')),
        ('transport_does_not_call_training_handler', '_handle_transport_smoke' in handler and "return _handle_training(event)" not in handler[handler.index('def _handle_transport_smoke'):handler.index('def handler(event):')]),
        ('transport_does_not_call_preview_handler', "return _handle_preview(event)" not in handler[handler.index('def _handle_transport_smoke'):handler.index('def handler(event):')]),
        ('transport_reports_no_r2', "'r2_read_executed': False" in handler and "'r2_write_executed': False" in handler),
        ('transport_reports_no_model_load', "'model_loaded': False" in handler),
        ('startup_logs_sdk_version', "'runpod_worker_boot'" in handler and "'runpod_serverless_start_enter'" in handler),
        ('job_received_log_present', "'runpod_job_received'" in handler),
        ('serverless_start_preserved', "runpod.serverless.start({'handler': handler})" in handler),
        ('script_requires_exact_confirmation', 'TESTAR TRANSPORTE RUNPOD SEM TREINO D3.6H10' in script),
        ('script_uses_transport_contract', 'privacy-identity-lora-transport-smoke-v1' in script),
        ('script_never_submits_training_contract', 'privacy-identity-lora-training-v2' not in script),
        ('script_has_no_automatic_retry', 'automaticRetryCreated = $false' in script and '/retry/' not in script),
        ('script_can_cancel_only_own_job', '/cancel/$jobId' in script and 'Purge' not in script),
        ('script_validates_nonce', 'TRANSPORT_SMOKE_CORRELATION_MISMATCH' in script),
    ]
    return checks


def test_stage_2_2d3_6h10_static_contract():
    failed = [name for name, ok in evaluate() if not ok]
    assert not failed, failed


if __name__ == '__main__':
    checks = evaluate()
    failed = [name for name, ok in checks if not ok]
    print(json.dumps({
        'status': 'TRAINER_STAGE_2_2D3_6H10_RUNPOD_SDK_TRANSPORT_SMOKE_READY' if not failed else 'TRAINER_STAGE_2_2D3_6H10_RUNPOD_SDK_TRANSPORT_SMOKE_BLOCKED',
        'totalChecks': len(checks),
        'passedChecks': len(checks) - len(failed),
        'failedChecks': failed,
        'safety': {
            'networkCalled': False,
            'runPodCalled': False,
            'gpuStarted': False,
            'trainingStarted': False,
            'r2Called': False,
        },
    }, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failed else 0)
