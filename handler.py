from __future__ import annotations
import tempfile, uuid
from pathlib import Path
import runpod
from identity_worker.config import Settings
from identity_worker.contracts import CONTRACT_VERSION, parse_training_request
from identity_worker.dataset import materialize_dataset
from identity_worker.model_lock import materialize_model
from identity_worker.storage import client as r2_client, upload_private
from identity_worker.trainer import build_command, run_training
from identity_worker.telemetry import log_event
from identity_worker.errors import WorkerError

settings=Settings()

def handler(event):
    request=parse_training_request(event)
    settings.validate_runtime()
    if request.output_bucket != settings.r2_bucket_name:
        raise WorkerError('OUTPUT_BUCKET_MISMATCH', 'O destino não corresponde ao bucket privado configurado.')
    for sample in request.payload['dataset']['samples']:
        for field in ('video_source', 'reference_image_source'):
            if sample[field]['bucket'] != settings.r2_bucket_name:
                raise WorkerError('SOURCE_BUCKET_MISMATCH', 'O material não pertence ao bucket privado configurado.')
    log_event('identity_training_received',request_id=request.request_id,actor_profile_id=request.actor_profile_id,training_run_id=request.training_run_id)
    try:
        with tempfile.TemporaryDirectory(dir=str(settings.runtime_root),prefix=f"identity_{request.training_run_id}_") as temp:
            work=Path(temp); s3=r2_client(settings)
            dataset_root,metadata_path=materialize_dataset(request,settings,work,s3)
            model_paths=materialize_model(request,settings)
            output_dir=work/'output'
            adapter=run_training(build_command(request,settings,dataset_root,metadata_path,model_paths,output_dir),output_dir)
            key=f"{request.output_prefix.rstrip('/')}/{uuid.uuid4().hex}/{adapter.name}"
            uploaded=upload_private(s3,adapter,request.output_bucket,key,{'private':'true','qa_required':'true','actor_profile_id':request.actor_profile_id,'training_run_id':request.training_run_id})
            return {'contract_version':CONTRACT_VERSION,'status':'training_completed','adapter':{
              **uploaded,'actor_profile_id':request.actor_profile_id,'training_run_id':request.training_run_id,
              'base_model_fingerprint':request.payload['model']['fingerprint_sha256'],'rank':request.payload['training']['lora_rank'],
              'alpha':request.payload['training']['lora_alpha'],'recommended_strength_model':0.65,'consent_version':'identity-preparation-v1',
              'manifest':{'dataset_manifest_sha256':request.payload['dataset_manifest_sha256'],'model_revision':request.payload['model']['revision'],'training_profile':request.payload['training']['profile']}}}
    except WorkerError as error:
        log_event('identity_training_failed',request_id=request.request_id,error_code=error.code,retryable=error.retryable)
        raise RuntimeError(f'{error.code}: {error}') from error

runpod.serverless.start({'handler':handler})
