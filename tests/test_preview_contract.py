from datetime import datetime, timedelta, timezone
import pytest
from identity_worker.contracts import parse_preview_request
from identity_worker.errors import WorkerError

ACTOR_ID='11111111-1111-4111-8111-111111111111'
RUN_ID='22222222-2222-4222-8222-222222222222'
ADAPTER_ID='33333333-3333-4333-8333-333333333333'


def preview_event():
    expires=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    return {'input':{
        'contract_version':'privacy-identity-lora-review-preview-v1',
        'execution_mode':'controlled_review_preview_smoke',
        'request_id':'preview-request',
        'actor_profile_id':ACTOR_ID,
        'training_run_id':RUN_ID,
        'adapter_id':ADAPTER_ID,
        'adapter':{'bucket':'privacy-media','key':f'identity-adapters/{ACTOR_ID}/{RUN_ID}/a.safetensors','sha256':'a'*64,'byte_size':100},
        'source':{
            'control_video':{'bucket':'privacy-media','key':'vault/actor-mapping/a/control.mp4','sha256':'b'*64},
            'reference_image':{'bucket':'privacy-media','key':'vault/actor-mapping/a/reference.jpg','sha256':'c'*64},
        },
        'model':{'repository':'Wan-AI/Wan2.1-VACE-14B','revision':'rev','fingerprint_sha256':'d'*64,'artifacts':[{'path':str(i),'sha256':'e'*64,'size':1} for i in range(9)]},
        'preview':{'profile':'identity_private_review_preview_v1','prompt':'identity person','negative_prompt':'bad','width':832,'height':480,'num_frames':17,'fps':15,'num_inference_steps':20,'seed':20260724,'lora_strength':0.65,'vace_scale':1.0,'one_output':True},
        'output':{'bucket':'privacy-media','prefix':f'identity-review-previews/{ACTOR_ID}/{RUN_ID}/{ADAPTER_ID}','public':False,'content_type':'video/mp4'},
        'smoke':{'enabled':True,'one_shot':True,'actor_profile_id':ACTOR_ID,'training_run_id':RUN_ID,'adapter_id':ADAPTER_ID,'expires_at':expires,'max_jobs':1},
        'safety':{'actor_scoped':True,'run_scoped':True,'adapter_scoped':True,'private_storage_only':True,'public_urls_forbidden':True,'product_release_allowed':False,'automatic_retry_allowed':False,'one_shot_smoke':True,'approval_allowed':False},
    }}


def test_preview_contract_accepts_private_one_shot_scope():
    request=parse_preview_request(preview_event())
    assert request.adapter_id == ADAPTER_ID
    assert request.output_bucket == 'privacy-media'


def test_preview_contract_rejects_public_adapter():
    event=preview_event(); event['input']['adapter']['key']='https://example.com/adapter.safetensors'
    with pytest.raises(WorkerError) as error: parse_preview_request(event)
    assert error.value.code == 'INVALID_PRIVATE_ADAPTER'


def test_preview_contract_rejects_scope_mismatch():
    event=preview_event(); event['input']['smoke']['adapter_id']='44444444-4444-4444-8444-444444444444'
    with pytest.raises(WorkerError) as error: parse_preview_request(event)
    assert error.value.code == 'PREVIEW_SCOPE_MISMATCH'


def test_preview_contract_rejects_long_or_multiple_profile():
    event=preview_event(); event['input']['preview']['num_frames']=49
    with pytest.raises(WorkerError) as error: parse_preview_request(event)
    assert error.value.code == 'INVALID_PREVIEW_PROFILE'


def test_preview_contract_rejects_approval_permission():
    event=preview_event(); event['input']['safety']['approval_allowed']=True
    with pytest.raises(WorkerError) as error: parse_preview_request(event)
    assert error.value.code == 'INVALID_PREVIEW_SAFETY'
