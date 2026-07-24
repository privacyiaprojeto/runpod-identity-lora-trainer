from datetime import datetime, timedelta, timezone
import pytest
from identity_worker.contracts import parse_preview_request
from identity_worker.errors import WorkerError

ACTOR_ID='11111111-1111-4111-8111-111111111111'
RUN_ID='22222222-2222-4222-8222-222222222222'
ADAPTER_ID='33333333-3333-4333-8333-333333333333'


def preview_event():
    expires=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    common_negative='identity mismatch, deformed face, deformed hands, blur, low resolution, text, watermark'
    return {'input':{
        'contract_version':'privacy-identity-lora-review-kit-v2',
        'execution_mode':'controlled_identity_qa_kit',
        'request_id':'preview-request','actor_profile_id':ACTOR_ID,'training_run_id':RUN_ID,'adapter_id':ADAPTER_ID,
        'adapter':{'bucket':'privacy-media','key':f'identity-adapters/{ACTOR_ID}/{RUN_ID}/a.safetensors','sha256':'a'*64,'byte_size':100},
        'source':{'control_video':{'bucket':'privacy-media','key':'vault/a/control.mp4','sha256':'b'*64},'reference_image':{'bucket':'privacy-media','key':'vault/a/reference.jpg','sha256':'c'*64}},
        'model':{'repository':'Wan-AI/Wan2.1-VACE-14B','revision':'rev','fingerprint_sha256':'d'*64,'artifacts':[{'path':str(i),'sha256':'e'*64,'size':1} for i in range(9)]},
        'preview':{
            'profile':'identity_private_qa_kit_v2','one_qa_kit':True,'asset_count':4,'lora_strength':0.65,
            'video':{'asset_key':'video_walk_turn_smile','label':'Caminhada, giro e sorriso','prompt':'adult person walking','negative_prompt':common_negative,'width':1024,'height':576,'num_frames':61,'fps':12,'num_inference_steps':24,'seed':20260724,'vace_scale':1.0},
            'images':[
                {'asset_key':'image_crying','label':'Expressão de choro','prompt':'adult portrait crying','negative_prompt':common_negative,'width':768,'height':1024,'num_frames':1,'num_inference_steps':28,'seed':20260725,'vace_scale':1.0},
                {'asset_key':'image_sensual','label':'Expressão sensual','prompt':'adult elegant portrait','negative_prompt':common_negative,'width':768,'height':1024,'num_frames':1,'num_inference_steps':28,'seed':20260726,'vace_scale':1.0},
                {'asset_key':'image_lollipop','label':'Interação com pirulito','prompt':'adult portrait with lollipop','negative_prompt':common_negative,'width':768,'height':1024,'num_frames':1,'num_inference_steps':28,'seed':20260727,'vace_scale':1.0},
            ],
        },
        'output':{'bucket':'privacy-media','prefix':f'identity-review-previews/{ACTOR_ID}/{RUN_ID}/{ADAPTER_ID}','public':False},
        'smoke':{'enabled':True,'one_shot':True,'actor_profile_id':ACTOR_ID,'training_run_id':RUN_ID,'adapter_id':ADAPTER_ID,'expires_at':expires,'max_jobs':1},
        'safety':{'actor_scoped':True,'run_scoped':True,'adapter_scoped':True,'private_storage_only':True,'public_urls_forbidden':True,'product_release_allowed':False,'automatic_retry_allowed':False,'one_shot_smoke':True,'approval_allowed':False,'qa_kit_only':True},
    }}


def test_preview_contract_accepts_universal_private_qa_kit():
    request=parse_preview_request(preview_event())
    assert request.adapter_id == ADAPTER_ID


def test_preview_contract_rejects_short_video():
    event=preview_event(); event['input']['preview']['video']['num_frames']=17
    with pytest.raises(WorkerError) as error: parse_preview_request(event)
    assert error.value.code == 'INVALID_PREVIEW_VIDEO_PROFILE'


def test_preview_contract_rejects_missing_image():
    event=preview_event(); event['input']['preview']['images'].pop()
    with pytest.raises(WorkerError) as error: parse_preview_request(event)
    assert error.value.code == 'INVALID_PREVIEW_IMAGE_SET'


def test_preview_contract_rejects_public_adapter():
    event=preview_event(); event['input']['adapter']['key']='https://example.com/adapter.safetensors'
    with pytest.raises(WorkerError) as error: parse_preview_request(event)
    assert error.value.code == 'INVALID_PRIVATE_ADAPTER'


def test_preview_contract_rejects_approval_permission():
    event=preview_event(); event['input']['safety']['approval_allowed']=True
    with pytest.raises(WorkerError) as error: parse_preview_request(event)
    assert error.value.code == 'INVALID_PREVIEW_SAFETY'
