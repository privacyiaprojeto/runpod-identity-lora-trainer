from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[1]
checks = []

def check(name, condition):
    checks.append({"name": name, "ok": bool(condition)})

contracts = (root / 'identity_worker' / 'contracts.py').read_text(encoding='utf-8')
check('contracts profile aligned', "DIT_TRAINING_PROFILE = 'wan_dit_identity_video_v1'" in contracts)

submit = (root / 'scripts' / 'poc' / 'SUBMIT_WAN_DIT_TRAINING_POC.ps1').read_text(encoding='utf-8')
check('submit validates approved profile', "wan_dit_identity_video_v1" in submit)
check('submit reads canonical endpoint env', "IDENTITY_LORA_TRAINER_ENDPOINT_ID" in submit)

payload = json.loads((root / 'poc' / 'training_input.local.example.json').read_text(encoding='utf-8'))
artifacts = payload['input']['model']['artifacts']
paths = [item['path'] for item in artifacts]
check('example payload has 9 artifacts', len(artifacts) == 9)
check('example payload uses 7 diffusion shards', any('00007-of-00007' in item for item in paths))
check('example payload drops configuration.json placeholder', not any('configuration.json' in item for item in paths))
check('example payload profile aligned', payload['input']['training']['profile'] == 'wan_dit_identity_video_v1')

failed = [item for item in checks if not item['ok']]
print(json.dumps({
    'status': 'TRAINER_STAGE_2_2D3_6H7_PATCH_READY' if not failed else 'TRAINER_STAGE_2_2D3_6H7_PATCH_NOT_READY',
    'totalChecks': len(checks),
    'passedChecks': len(checks) - len(failed),
    'failedChecks': failed,
}, indent=2))
if failed:
    sys.exit(1)
