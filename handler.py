from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import runpod

from identity_worker.config import Settings
from identity_worker.contracts import CONTRACT_VERSION, parse_training_request
from identity_worker.dataset import materialize_dataset
from identity_worker.errors import WorkerError
from identity_worker.model_lock import materialize_model
from identity_worker.model_preflight import assert_model_binding_compatible
from identity_worker.one_shot import reserve_one_shot, update_one_shot
from identity_worker.runtime_preflight import assert_runtime_compatible
from identity_worker.storage import client as r2_client, upload_private
from identity_worker.telemetry import log_event
from identity_worker.trainer import build_command, run_training

settings = Settings()


def _assert_worker_scope(request) -> None:
    if request.actor_profile_id != settings.smoke_actor_profile_id:
        raise WorkerError("WORKER_ACTOR_SCOPE_MISMATCH", "O worker não está autorizado para este ator.")
    if request.training_run_id != settings.smoke_training_run_id:
        raise WorkerError("WORKER_RUN_SCOPE_MISMATCH", "O worker não está autorizado para este run.")
    configured_expiry = settings.smoke_expiry()
    if not configured_expiry or request.smoke_expires_at != configured_expiry.isoformat():
        raise WorkerError("WORKER_SMOKE_EXPIRY_MISMATCH", "A janela do contrato não corresponde à janela configurada no worker.")


def handler(event):
    lock_path: Path | None = None
    request = None
    try:
        request = parse_training_request(event)
        settings.validate_runtime()
        _assert_worker_scope(request)
        if request.output_bucket != settings.r2_bucket_name:
            raise WorkerError("OUTPUT_BUCKET_MISMATCH", "O destino não corresponde ao bucket privado configurado.")
        for sample in request.payload["dataset"]["samples"]:
            for field in ("video_source", "reference_image_source"):
                if sample[field]["bucket"] != settings.r2_bucket_name:
                    raise WorkerError("SOURCE_BUCKET_MISMATCH", "O material não pertence ao bucket privado configurado.")

        # Runtime and request-specific model binding must pass before the one-shot lock is consumed.
        runtime = assert_runtime_compatible(settings.diffsynth_root)
        log_event(
            "identity_training_runtime_ready",
            request_id=request.request_id,
            versions=runtime["versions"],
        )
        model_binding = materialize_model(request, settings)
        model_probe = assert_model_binding_compatible(model_binding)
        log_event(
            "identity_training_model_binding_ready",
            request_id=request.request_id,
            model_name=model_probe["modelName"],
            model_hash=model_probe["modelHash"],
            diffusion_shard_count=model_probe["diffusionShardCount"],
        )

        lock_path = reserve_one_shot(
            settings.smoke_lock_root,
            request.actor_profile_id,
            request.training_run_id,
            request.request_id,
        )
        log_event(
            "identity_training_smoke_reserved",
            request_id=request.request_id,
            actor_profile_id=request.actor_profile_id,
            training_run_id=request.training_run_id,
        )

        with tempfile.TemporaryDirectory(dir=str(settings.runtime_root), prefix=f"identity_{request.training_run_id}_") as temp:
            work = Path(temp)
            s3 = r2_client(settings)
            update_one_shot(lock_path, "materializing_dataset")
            dataset_root, metadata_path = materialize_dataset(request, settings, work, s3)
            output_dir = work / "output"
            update_one_shot(lock_path, "training")
            adapter = run_training(
                build_command(request, settings, dataset_root, metadata_path, model_binding, output_dir),
                output_dir,
            )
            key = f"{request.output_prefix.rstrip('/')}/{uuid.uuid4().hex}/{adapter.name}"
            update_one_shot(lock_path, "uploading_adapter")
            uploaded = upload_private(
                s3,
                adapter,
                request.output_bucket,
                key,
                {'private':'true','qa_required':'true','actor_profile_id':request.actor_profile_id,'training_run_id':request.training_run_id,'one_shot_smoke':'true'},
            )
            update_one_shot(lock_path, 'completed', adapterSha256=uploaded["sha256"], adapterKey=uploaded["r2_key"])
            return {
                "contract_version": CONTRACT_VERSION,
                "status": "training_completed",
                "adapter": {
                    **uploaded,
                    "actor_profile_id": request.actor_profile_id,
                    "training_run_id": request.training_run_id,
                    "base_model_fingerprint": request.payload["model"]["fingerprint_sha256"],
                    "rank": request.payload["training"]["lora_rank"],
                    "alpha": request.payload["training"]["lora_alpha"],
                    "recommended_strength_model": 0.65,
                    "consent_version": "identity-preparation-v1",
                    "manifest": {
                        "dataset_manifest_sha256": request.payload["dataset_manifest_sha256"],
                        "model_revision": request.payload["model"]["revision"],
                        "training_profile": request.payload["training"]["profile"],
                        "one_shot_smoke": True,
                        "grouped_model_binding": True,
                    },
                },
            }
    except WorkerError as error:
        if lock_path is not None:
            update_one_shot(lock_path, 'failed', errorCode=error.code, retryable=error.retryable)
        log_event(
            "identity_training_failed",
            request_id=getattr(request, "request_id", None),
            error_code=error.code,
            retryable=error.retryable,
            one_shot_reserved=lock_path is not None,
        )
        raise RuntimeError(f"{error.code}: {error}") from error
    except Exception as error:
        if lock_path is not None:
            update_one_shot(lock_path, 'failed', errorCode=type(error).__name__, retryable=False)
        log_event(
            "identity_training_failed",
            request_id=getattr(request, "request_id", None),
            error_code=type(error).__name__,
            retryable=False,
            one_shot_reserved=lock_path is not None,
        )
        raise


runpod.serverless.start({"handler": handler})
