from __future__ import annotations

import os
import platform
import tempfile
import time
import uuid
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import runpod

from identity_worker.config import Settings
from identity_worker.contracts import (
    CONTRACT_VERSION,
    PREVIEW_CONTRACT_VERSION,
    R2_PREFLIGHT_CONTRACT_VERSION,
    parse_preview_request,
    parse_r2_preflight_request,
    parse_training_request,
)
from identity_worker.dataset import materialize_dataset
from identity_worker.errors import WorkerError
from identity_worker.model_lock import materialize_model
from identity_worker.model_preflight import assert_model_binding_compatible
from identity_worker.one_shot import reserve_one_shot, reserve_preview_one_shot, update_one_shot
from identity_worker.preview import materialize_preview_inputs, run_qa_kit
from identity_worker.runtime_preflight import assert_runtime_compatible
from identity_worker.r2_preflight import probe_private_r2_metadata
from identity_worker.storage import client as r2_client, upload_private
from identity_worker.telemetry import log_event
from identity_worker.adapter_scope import collect_and_audit_checkpoints
from identity_worker.trainer import build_command, run_training

settings = Settings()


TRANSPORT_SMOKE_CONTRACT_VERSION = 'privacy-identity-lora-transport-smoke-v1'


def _package_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return 'unknown'


RUNPOD_SDK_VERSION = _package_version('runpod')
TRANSPORT_SMOKE_ENABLED = os.getenv('PRIVACY_LORA_TRANSPORT_SMOKE_ENABLED', 'false').strip().lower() == 'true'


log_event(
    'runpod_worker_boot',
    runpod_sdk_version=RUNPOD_SDK_VERSION,
    python_version=platform.python_version(),
    transport_smoke_enabled=TRANSPORT_SMOKE_ENABLED,
    training_enabled=settings.allow_training,
    training_dry_run_only=settings.dry_run_only,
    training_smoke_mode=settings.smoke_mode,
    preview_enabled=settings.allow_preview,
    preview_mode=settings.preview_mode,
    r2_preflight_enabled=settings.r2_preflight_enabled,
)


def _input_payload(event):
    return event.get('input') if isinstance(event.get('input'), dict) else event


def _assert_training_scope(request) -> None:
    if request.actor_profile_id != settings.smoke_actor_profile_id:
        raise WorkerError('WORKER_ACTOR_SCOPE_MISMATCH', 'O worker não está autorizado para este ator.')
    if request.training_run_id != settings.smoke_training_run_id:
        raise WorkerError('WORKER_RUN_SCOPE_MISMATCH', 'O worker não está autorizado para este run.')
    configured_expiry = settings.smoke_expiry()
    if not configured_expiry or request.smoke_expires_at != configured_expiry.isoformat():
        raise WorkerError('WORKER_SMOKE_EXPIRY_MISMATCH', 'A janela do contrato não corresponde à janela configurada no worker.')


def _assert_r2_preflight_scope(request) -> None:
    if request.actor_profile_id != settings.r2_preflight_actor_profile_id:
        raise WorkerError('R2_PREFLIGHT_ACTOR_SCOPE_MISMATCH', 'O worker não está autorizado para este ator no preflight do R2.')
    if request.training_run_id != settings.r2_preflight_training_run_id:
        raise WorkerError('R2_PREFLIGHT_RUN_SCOPE_MISMATCH', 'O worker não está autorizado para este run no preflight do R2.')
    configured_expiry = settings.r2_preflight_expiry()
    if not configured_expiry or request.smoke_expires_at != configured_expiry.isoformat():
        raise WorkerError('R2_PREFLIGHT_EXPIRY_MISMATCH', 'A janela do preflight não corresponde à configuração do worker.')


def _assert_preview_scope(request) -> None:
    if request.actor_profile_id != settings.preview_actor_profile_id:
        raise WorkerError('PREVIEW_ACTOR_SCOPE_MISMATCH', 'O worker não está autorizado para este ator.')
    if request.training_run_id != settings.preview_training_run_id:
        raise WorkerError('PREVIEW_RUN_SCOPE_MISMATCH', 'O worker não está autorizado para este run.')
    if request.adapter_id != settings.preview_adapter_id:
        raise WorkerError('PREVIEW_ADAPTER_SCOPE_MISMATCH', 'O worker não está autorizado para este adapter.')
    configured_expiry = settings.preview_expiry()
    if not configured_expiry or request.smoke_expires_at != configured_expiry.isoformat():
        raise WorkerError('PREVIEW_EXPIRY_MISMATCH', 'A janela da prévia não corresponde à configuração do worker.')


def _handle_training(event):
    lock_path: Path | None = None
    request = None
    try:
        request = parse_training_request(event)
        settings.validate_runtime()
        _assert_training_scope(request)
        if request.output_bucket != settings.r2_bucket_name:
            raise WorkerError('OUTPUT_BUCKET_MISMATCH', 'O destino não corresponde ao bucket privado configurado.')
        for sample in request.payload['dataset']['samples']:
            for field in ('video_source', 'reference_image_source'):
                if sample[field]['bucket'] != settings.r2_bucket_name:
                    raise WorkerError('SOURCE_BUCKET_MISMATCH', 'O material não pertence ao bucket privado configurado.')

        runtime = assert_runtime_compatible(settings.diffsynth_root)
        log_event('identity_training_runtime_ready', request_id=request.request_id, versions=runtime['versions'])
        model_binding = materialize_model(request, settings)
        model_probe = assert_model_binding_compatible(model_binding)
        log_event('identity_training_model_binding_ready', request_id=request.request_id, model_name=model_probe['modelName'], model_hash=model_probe['modelHash'], diffusion_shard_count=model_probe['diffusionShardCount'])

        lock_path = reserve_one_shot(settings.smoke_lock_root, request.actor_profile_id, request.training_run_id, request.request_id)
        log_event('identity_training_smoke_reserved', request_id=request.request_id, actor_profile_id=request.actor_profile_id, training_run_id=request.training_run_id)

        with tempfile.TemporaryDirectory(dir=str(settings.runtime_root), prefix=f'identity_{request.training_run_id}_') as temp:
            work = Path(temp)
            s3 = r2_client(settings)
            update_one_shot(lock_path, 'materializing_dataset')
            dataset_root, metadata_path = materialize_dataset(request, settings, work, s3)
            output_dir = work / 'output'
            update_one_shot(lock_path, 'training')
            adapter = run_training(build_command(request, settings, dataset_root, metadata_path, model_binding, output_dir), output_dir)
            checkpoints = collect_and_audit_checkpoints(output_dir)
            batch_id = uuid.uuid4().hex
            update_one_shot(lock_path, 'uploading_adapter')
            uploaded_checkpoints = []
            for checkpoint in checkpoints:
                checkpoint_path = checkpoint['path']
                checkpoint_step = checkpoint['step']
                key = f"{request.output_prefix.rstrip('/')}/{batch_id}/step-{checkpoint_step}.safetensors"
                uploaded_checkpoint = upload_private(
                    s3, checkpoint_path, request.output_bucket, key,
                    {'private':'true','qa_required':'true','actor_profile_id':request.actor_profile_id,'training_run_id':request.training_run_id,'one_shot_smoke':'true','lora_base_model':'dit','checkpoint_step':str(checkpoint_step)},
                )
                uploaded_checkpoints.append({**uploaded_checkpoint, 'step': checkpoint_step, 'tensor_count': checkpoint['tensor_count']})
            uploaded = next(item for item in uploaded_checkpoints if item['step'] == 800)
            update_one_shot(lock_path, 'completed', adapterSha256=uploaded['sha256'], adapterKey=uploaded['r2_key'], checkpointCount=len(uploaded_checkpoints))
            return {
                'contract_version': CONTRACT_VERSION,
                'status': 'training_completed',
                'adapter': {
                    **uploaded,
                    'actor_profile_id': request.actor_profile_id,
                    'training_run_id': request.training_run_id,
                    'base_model_fingerprint': request.payload['model']['fingerprint_sha256'],
                    'rank': request.payload['training']['lora_rank'],
                    'alpha': request.payload['training']['lora_alpha'],
                    'recommended_strength_model': 0.65,
                    'consent_version': 'identity-preparation-v1',
                    'manifest': {
                        'dataset_manifest_sha256': request.payload['dataset_manifest_sha256'],
                        'model_revision': request.payload['model']['revision'],
                        'training_profile': request.payload['training']['profile'],
                        'one_shot_smoke': True,
                        'grouped_model_binding': True,
                        'lora_base_model': 'dit',
                        'remove_prefix_in_ckpt': 'pipe.dit.',
                        'target_modules': request.payload['training']['target_modules'],
                        'optimizer_steps': 800,
                        'checkpoints': uploaded_checkpoints,
                    },
                },
            }
    except WorkerError as error:
        if lock_path is not None:
            update_one_shot(lock_path, 'failed', errorCode=error.code, retryable=error.retryable)
        log_event('identity_training_failed', request_id=getattr(request, 'request_id', None), error_code=error.code, retryable=error.retryable, one_shot_reserved=lock_path is not None)
        raise RuntimeError(f'{error.code}: {error}') from error
    except Exception as error:
        if lock_path is not None:
            update_one_shot(lock_path, 'failed', errorCode=type(error).__name__, retryable=False)
        log_event('identity_training_failed', request_id=getattr(request, 'request_id', None), error_code=type(error).__name__, retryable=False, one_shot_reserved=lock_path is not None)
        raise


def _handle_preview(event):
    lock_path: Path | None = None
    request = None
    try:
        request = parse_preview_request(event)
        settings.validate_preview_runtime()
        _assert_preview_scope(request)
        if request.output_bucket != settings.r2_bucket_name:
            raise WorkerError('PREVIEW_OUTPUT_BUCKET_MISMATCH', 'O destino do kit não corresponde ao bucket privado.')
        for item in (request.payload['adapter'], request.payload['source']['control_video'], request.payload['source']['reference_image']):
            if item['bucket'] != settings.r2_bucket_name:
                raise WorkerError('PREVIEW_SOURCE_BUCKET_MISMATCH', 'O kit contém referência fora do bucket privado autorizado.')

        runtime = assert_runtime_compatible(settings.diffsynth_root)
        log_event('identity_qa_kit_runtime_ready', request_id=request.request_id, versions=runtime['versions'])
        model_binding = materialize_model(request, settings)
        model_probe = assert_model_binding_compatible(model_binding)
        log_event('identity_qa_kit_model_ready', request_id=request.request_id, model_name=model_probe['modelName'], model_hash=model_probe['modelHash'])

        lock_path = reserve_preview_one_shot(
            settings.preview_lock_root, request.actor_profile_id, request.training_run_id, request.adapter_id, request.request_id,
            recovery_enabled=settings.preview_recovery_enabled,
            recovery_required_error_code=settings.preview_recovery_required_error_code,
            recovery_max_attempts=settings.preview_recovery_max_attempts,
            replacement_enabled=settings.preview_replacement_enabled,
            replacement_required_reason=settings.preview_replacement_required_reason,
            replacement_max_attempts=settings.preview_replacement_max_attempts,
        )
        with tempfile.TemporaryDirectory(dir=str(settings.runtime_root), prefix=f'preview_{request.adapter_id}_') as temp:
            work = Path(temp)
            s3 = r2_client(settings)
            update_one_shot(lock_path, 'materializing_preview_inputs')
            inputs = materialize_preview_inputs(request, work, s3)
            update_one_shot(lock_path, 'generating_qa_kit')
            generated_assets = run_qa_kit(request, settings, model_binding, inputs, work)
            update_one_shot(lock_path, 'uploading_qa_kit')
            uploaded_assets = []
            batch_id = uuid.uuid4().hex
            for asset_key, item in generated_assets.items():
                extension = '.mp4' if item['kind'] == 'video' else '.png'
                key = f"{request.output_prefix.rstrip('/')}/{batch_id}/{asset_key}{extension}"
                uploaded = upload_private(
                    s3, item['path'], request.output_bucket, key,
                    {'private':'true','qa_only':'true','qa_kit':'true','asset_key':asset_key,'actor_profile_id':request.actor_profile_id,'training_run_id':request.training_run_id,'adapter_id':request.adapter_id,'one_shot_preview':'true'},
                    content_type=item['content_type'],
                )
                uploaded_assets.append({
                    **uploaded, 'asset_key': asset_key, 'label': item['label'], 'kind': item['kind'],
                    'content_type': item['content_type'], 'width': item['width'], 'height': item['height'],
                    'num_frames': item['num_frames'], 'fps': item['fps'], 'duration_seconds': item['duration_seconds'],
                    'private_only': True, 'approval_allowed': False,
                })
            video_asset = next(item for item in uploaded_assets if item['asset_key'] == 'video_walk_turn_smile')
            update_one_shot(lock_path, 'completed', qaKitAssetCount=len(uploaded_assets), previewSha256=video_asset['sha256'], previewKey=video_asset['r2_key'], previewDurationSeconds=video_asset['duration_seconds'])
            return {
                'contract_version': PREVIEW_CONTRACT_VERSION,
                'status': 'qa_kit_completed',
                'qa_kit': {
                    'schema_version': 'privacy-identity-qa-kit-v1',
                    'qa_kit_id': str(uuid.uuid4()),
                    'actor_profile_id': request.actor_profile_id,
                    'training_run_id': request.training_run_id,
                    'adapter_id': request.adapter_id,
                    'asset_count': len(uploaded_assets),
                    'assets': uploaded_assets,
                    'reviewable': video_asset['duration_seconds'] >= 4 and video_asset['num_frames'] >= 49,
                    'private_only': True,
                    'approval_allowed': False,
                },
            }
    except WorkerError as error:
        if lock_path is not None:
            update_one_shot(lock_path, 'failed', errorCode=error.code, retryable=error.retryable)
        log_event('identity_qa_kit_failed', request_id=getattr(request, 'request_id', None), error_code=error.code, retryable=error.retryable, one_shot_reserved=lock_path is not None)
        raise RuntimeError(f'{error.code}: {error}') from error
    except Exception as error:
        if lock_path is not None:
            update_one_shot(lock_path, 'failed', errorCode=type(error).__name__, retryable=False)
        log_event('identity_qa_kit_failed', request_id=getattr(request, 'request_id', None), error_code=type(error).__name__, retryable=False, one_shot_reserved=lock_path is not None)
        raise

def _handle_r2_preflight(event):
    request = None
    try:
        request = parse_r2_preflight_request(event)
        settings.validate_r2_preflight_runtime()
        if TRANSPORT_SMOKE_ENABLED:
            raise WorkerError('R2_PREFLIGHT_REQUIRES_TRANSPORT_CLOSED', 'Feche o transport smoke antes do preflight privado do R2.')
        _assert_r2_preflight_scope(request)
        if request.bucket != settings.r2_bucket_name:
            raise WorkerError('R2_PREFLIGHT_CONFIGURED_BUCKET_MISMATCH', 'O bucket do contrato não corresponde ao bucket privado configurado.')

        s3 = r2_client(settings)
        result = probe_private_r2_metadata(s3, request.bucket, request.objects)
        log_event(
            'identity_r2_private_preflight_completed',
            request_id=request.request_id,
            actor_profile_id=request.actor_profile_id,
            training_run_id=request.training_run_id,
            dataset_manifest_prefix=request.dataset_manifest_sha256[:12],
            objects_checked=result['objects_checked'],
        )
        return {
            'contract_version': R2_PREFLIGHT_CONTRACT_VERSION,
            'status': 'r2_private_preflight_completed',
            'request_id': request.request_id,
            'actor_profile_id': request.actor_profile_id,
            'training_run_id': request.training_run_id,
            'dataset_manifest_prefix': request.dataset_manifest_sha256[:12],
            'storage': result,
            'runpod_sdk_version': RUNPOD_SDK_VERSION,
            'worker_pid': os.getpid(),
            'timestamp_ms': int(time.time() * 1000),
            'safety': {
                'training_started': False,
                'preview_started': False,
                'model_loaded': False,
                'r2_metadata_read_executed': True,
                'r2_object_download_executed': False,
                'r2_write_executed': False,
                'r2_delete_executed': False,
                'r2_list_executed': False,
                'automatic_retry_created': False,
            },
        }
    except WorkerError as error:
        log_event(
            'identity_r2_private_preflight_failed',
            request_id=getattr(request, 'request_id', None),
            error_code=error.code,
            retryable=error.retryable,
        )
        raise RuntimeError(f'{error.code}: {error}') from error


def _handle_transport_smoke(event):
    payload = _input_payload(event)
    if not TRANSPORT_SMOKE_ENABLED:
        raise RuntimeError('TRANSPORT_SMOKE_DISABLED: o diagnóstico de transporte permanece fechado.')
    if settings.allow_training or not settings.dry_run_only or settings.smoke_mode:
        raise RuntimeError('TRANSPORT_SMOKE_REQUIRES_TRAINING_CLOSED: feche treinamento, dry-run e smoke antes do diagnóstico.')
    if settings.allow_preview or settings.preview_mode:
        raise RuntimeError('TRANSPORT_SMOKE_REQUIRES_PREVIEW_CLOSED: feche a prévia antes do diagnóstico.')
    if not isinstance(payload, dict):
        raise RuntimeError('TRANSPORT_SMOKE_PAYLOAD_INVALID: input deve ser um objeto.')

    execution_mode = str(payload.get('execution_mode') or '').strip()
    request_id = str(payload.get('request_id') or '').strip()
    nonce = str(payload.get('nonce') or '').strip()
    if execution_mode != 'queue_transport_only':
        raise RuntimeError('TRANSPORT_SMOKE_MODE_INVALID: use queue_transport_only.')
    if not request_id or len(request_id) > 128:
        raise RuntimeError('TRANSPORT_SMOKE_REQUEST_ID_INVALID')
    if len(nonce) < 16 or len(nonce) > 128:
        raise RuntimeError('TRANSPORT_SMOKE_NONCE_INVALID')

    log_event(
        'runpod_transport_smoke_received',
        request_id=request_id,
        provider_job_id=str(event.get('id') or '') if isinstance(event, dict) else '',
        runpod_sdk_version=RUNPOD_SDK_VERSION,
    )
    return {
        'contract_version': TRANSPORT_SMOKE_CONTRACT_VERSION,
        'status': 'transport_smoke_completed',
        'request_id': request_id,
        'nonce': nonce,
        'runpod_sdk_version': RUNPOD_SDK_VERSION,
        'worker_pid': os.getpid(),
        'timestamp_ms': int(time.time() * 1000),
        'safety': {
            'training_started': False,
            'preview_started': False,
            'r2_read_executed': False,
            'r2_write_executed': False,
            'model_loaded': False,
            'automatic_retry_created': False,
        },
    }


def handler(event):
    payload = _input_payload(event)
    contract_version = payload.get('contract_version') if isinstance(payload, dict) else None
    log_event(
        'runpod_job_received',
        provider_job_id=str(event.get('id') or '') if isinstance(event, dict) else '',
        contract_version=str(contract_version or ''),
    )
    if contract_version == TRANSPORT_SMOKE_CONTRACT_VERSION:
        return _handle_transport_smoke(event)
    if contract_version == R2_PREFLIGHT_CONTRACT_VERSION:
        return _handle_r2_preflight(event)
    if contract_version == PREVIEW_CONTRACT_VERSION:
        return _handle_preview(event)
    return _handle_training(event)


log_event(
    'runpod_serverless_start_enter',
    runpod_sdk_version=RUNPOD_SDK_VERSION,
    handler_name='handler',
    transport_contract_version=TRANSPORT_SMOKE_CONTRACT_VERSION,
    r2_preflight_contract_version=R2_PREFLIGHT_CONTRACT_VERSION,
)
runpod.serverless.start({'handler': handler})
