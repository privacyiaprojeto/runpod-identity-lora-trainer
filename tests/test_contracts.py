import pytest
from identity_worker.contracts import parse_training_request
from identity_worker.errors import WorkerError

def sample_event():
    sample={'video_source':{'bucket':'privacy-media','key':'vault/actor-mapping/a/v.mp4'},'video_sha256':'a'*64,'reference_image_source':{'bucket':'privacy-media','key':'vault/actor-mapping/a/i.jpg'},'reference_image_sha256':'b'*64}
    return {'input':{'contract_version':'privacy-identity-lora-training-v1','execution_mode':'controlled_training','request_id':'r','actor_profile_id':'11111111-1111-4111-8111-111111111111','training_run_id':'22222222-2222-4222-8222-222222222222','dataset_manifest_sha256':'c'*64,'dataset':{'samples':[dict(sample) for _ in range(15)]},'model':{'fingerprint_sha256':'d'*64,'artifacts':[{'path':str(i),'sha256':'e'*64,'size':1} for i in range(9)]},'output':{'bucket':'privacy-media','prefix':'identity-adapters/11111111-1111-4111-8111-111111111111/22222222-2222-4222-8222-222222222222','public':False},'safety':{'actor_scoped':True,'private_storage_only':True,'public_urls_forbidden':True,'product_release_allowed':False,'inference_injection_allowed':False,'automatic_retry_allowed':False}}}

def test_contract_accepts_private_scope():
    assert parse_training_request(sample_event()).actor_profile_id.startswith('1111')

def test_contract_rejects_public_url():
    event=sample_event(); event['input']['dataset']['samples'][0]['video_source']['key']='https://example.com/a.mp4'
    with pytest.raises(WorkerError): parse_training_request(event)
