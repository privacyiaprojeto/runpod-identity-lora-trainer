from datetime import datetime, timedelta, timezone
import pytest
from identity_worker.contracts import parse_training_request
from identity_worker.errors import WorkerError

ACTOR_ID='11111111-1111-4111-8111-111111111111'
RUN_ID='22222222-2222-4222-8222-222222222222'


def sample_event():
    sample={'video_source':{'bucket':'privacy-media','key':'vault/actor-mapping/a/v.mp4'},'video_sha256':'a'*64,'reference_image_source':{'bucket':'privacy-media','key':'vault/actor-mapping/a/i.jpg'},'reference_image_sha256':'b'*64}
    expires=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
    return {'input':{'contract_version':'privacy-identity-lora-training-v2','execution_mode':'controlled_training_smoke','request_id':'r','actor_profile_id':ACTOR_ID,'training_run_id':RUN_ID,'dataset_manifest_sha256':'c'*64,'dataset':{'samples':[dict(sample) for _ in range(15)]},'model':{'fingerprint_sha256':'d'*64,'artifacts':[{'path':str(i),'sha256':'e'*64,'size':1} for i in range(9)]},'output':{'bucket':'privacy-media','prefix':f'identity-adapters/{ACTOR_ID}/{RUN_ID}','public':False},'smoke':{'enabled':True,'one_shot':True,'actor_profile_id':ACTOR_ID,'training_run_id':RUN_ID,'expires_at':expires,'max_jobs':1},'safety':{'actor_scoped':True,'private_storage_only':True,'public_urls_forbidden':True,'product_release_allowed':False,'inference_injection_allowed':False,'automatic_retry_allowed':False,'one_shot_smoke':True}}}


def test_contract_accepts_private_scope():
    assert parse_training_request(sample_event()).actor_profile_id == ACTOR_ID


def test_contract_rejects_public_url():
    event=sample_event(); event['input']['dataset']['samples'][0]['video_source']['key']='https://example.com/a.mp4'
    with pytest.raises(WorkerError): parse_training_request(event)


def test_contract_rejects_wrong_smoke_scope():
    event=sample_event(); event['input']['smoke']['training_run_id']='33333333-3333-4333-8333-333333333333'
    with pytest.raises(WorkerError) as error: parse_training_request(event)
    assert error.value.code == 'SMOKE_SCOPE_MISMATCH'
