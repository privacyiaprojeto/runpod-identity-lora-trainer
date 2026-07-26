from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evaluate():
    handler = (ROOT / 'handler.py').read_text(encoding='utf-8')
    config = (ROOT / 'identity_worker' / 'config.py').read_text(encoding='utf-8')
    contracts = (ROOT / 'identity_worker' / 'contracts.py').read_text(encoding='utf-8')
    probe = (ROOT / 'identity_worker' / 'r2_preflight.py').read_text(encoding='utf-8')
    script = (ROOT / 'scripts' / 'poc' / 'TEST_RUNPOD_R2_PREFLIGHT.ps1').read_text(encoding='utf-8')
    env_example = (ROOT / '.env.example').read_text(encoding='utf-8')

    handle_slice = handler[handler.index('def _handle_r2_preflight'):handler.index('def _handle_transport_smoke')]
    checks = [
        ('r2_preflight_contract_declared', 'privacy-identity-lora-r2-preflight-v1' in contracts),
        ('r2_preflight_disabled_by_default', "PRIVACY_LORA_R2_PREFLIGHT_ENABLED', False" in config),
        ('r2_preflight_has_dedicated_scope', 'PRIVACY_LORA_R2_PREFLIGHT_ACTOR_PROFILE_ID' in config and 'PRIVACY_LORA_R2_PREFLIGHT_TRAINING_RUN_ID' in config and 'PRIVACY_LORA_R2_PREFLIGHT_EXPIRES_AT' in config),
        ('r2_preflight_requires_training_closed', 'R2_PREFLIGHT_REQUIRES_TRAINING_CLOSED' in config),
        ('r2_preflight_requires_preview_closed', 'R2_PREFLIGHT_REQUIRES_PREVIEW_CLOSED' in config),
        ('r2_preflight_requires_transport_closed', 'R2_PREFLIGHT_REQUIRES_TRANSPORT_CLOSED' in handler),
        ('r2_preflight_branches_before_training', handler.index('contract_version == R2_PREFLIGHT_CONTRACT_VERSION') < handler.index('return _handle_training(event)')),
        ('r2_preflight_does_not_call_training', 'run_training(' not in handle_slice and 'materialize_model(' not in handle_slice and 'materialize_dataset(' not in handle_slice),
        ('r2_preflight_does_not_call_preview', '_handle_preview(' not in handle_slice and 'run_qa_kit(' not in handle_slice),
        ('r2_probe_uses_head_object', 'head_object(' in probe),
        ('r2_probe_never_downloads', 'download_file(' not in probe and 'get_object(' not in probe),
        ('r2_probe_never_writes', 'upload_file(' not in probe and 'put_object(' not in probe),
        ('r2_probe_never_deletes', 'delete_object(' not in probe),
        ('r2_probe_never_lists', 'list_objects' not in probe),
        ('r2_result_masks_bucket_and_keys', 'bucket_masked' in probe and 'key_fingerprint' in probe),
        ('r2_preflight_reports_no_model_or_training', "'training_started': False" in handle_slice and "'model_loaded': False" in handle_slice),
        ('r2_preflight_reports_no_write_delete_download', "'r2_object_download_executed': False" in handle_slice and "'r2_write_executed': False" in handle_slice and "'r2_delete_executed': False" in handle_slice),
        ('r2_preflight_has_exact_confirmation', 'TESTAR R2 PRIVADO SEM TREINO D3.6H11' in script),
        ('r2_preflight_script_has_no_retry', '/retry/' not in script and 'automaticRetryCreated = $false' in script),
        ('r2_preflight_script_can_cancel_only_own_job', '/cancel/$jobId' in script and 'Purge' not in script),
        ('r2_preflight_env_documented', 'PRIVACY_LORA_R2_PREFLIGHT_ENABLED=false' in env_example),
    ]
    return checks


def test_stage_2_2d3_6h11_private_r2_preflight_static():
    failed = [name for name, ok in evaluate() if not ok]
    assert not failed, failed


if __name__ == '__main__':
    checks = evaluate()
    failed = [name for name, ok in checks if not ok]
    print(json.dumps({
        'status': 'TRAINER_STAGE_2_2D3_6H11_PRIVATE_R2_PREFLIGHT_READY' if not failed else 'TRAINER_STAGE_2_2D3_6H11_PRIVATE_R2_PREFLIGHT_BLOCKED',
        'totalChecks': len(checks),
        'passedChecks': len(checks) - len(failed),
        'failedChecks': failed,
        'safety': {
            'networkCalled': False,
            'runPodCalled': False,
            'gpuStarted': False,
            'trainingStarted': False,
            'modelLoaded': False,
            'r2Called': False,
            'r2WriteCalled': False,
            'r2DeleteCalled': False,
        },
    }, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failed else 0)
