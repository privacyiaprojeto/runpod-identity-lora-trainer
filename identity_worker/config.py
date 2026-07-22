from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from .errors import WorkerError


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() == 'true'


def _text(name: str, default: str = '') -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    allow_training: bool = _bool('PRIVACY_LORA_ALLOW_TRAINING', False)
    dry_run_only: bool = _bool('PRIVACY_LORA_DRY_RUN_ONLY', True)
    smoke_mode: bool = _bool('PRIVACY_LORA_SMOKE_MODE', False)
    smoke_actor_profile_id: str = _text('PRIVACY_LORA_SMOKE_ACTOR_PROFILE_ID')
    smoke_training_run_id: str = _text('PRIVACY_LORA_SMOKE_TRAINING_RUN_ID')
    smoke_expires_at: str = _text('PRIVACY_LORA_SMOKE_EXPIRES_AT')
    r2_account_id: str = _text('R2_ACCOUNT_ID')
    r2_access_key_id: str = _text('R2_ACCESS_KEY_ID')
    r2_secret_access_key: str = _text('R2_SECRET_ACCESS_KEY')
    r2_bucket_name: str = _text('R2_BUCKET_NAME')
    hf_token: str = _text('HF_TOKEN')
    app_root: Path = Path(_text('APP_ROOT', '/app'))
    runtime_root: Path = Path(_text('RUNTIME_ROOT', '/runpod-volume/privacy-identity-lora'))
    model_cache_root: Path = Path(_text('MODEL_CACHE_ROOT', '/runpod-volume/models/identity-lora'))
    diffsynth_root: Path = Path(_text('DIFFSYNTH_ROOT', '/opt/DiffSynth-Studio'))

    @property
    def r2_endpoint_url(self) -> str:
        return f'https://{self.r2_account_id}.r2.cloudflarestorage.com'

    @property
    def smoke_lock_root(self) -> Path:
        return self.runtime_root / 'smoke-locks'

    def smoke_expiry(self) -> datetime | None:
        if not self.smoke_expires_at:
            return None
        try:
            value = self.smoke_expires_at.replace('Z', '+00:00')
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def validate_runtime(self) -> None:
        if self.dry_run_only or not self.allow_training:
            raise WorkerError('TRAINING_DISABLED', 'O treinamento real permanece desligado por política.')
        if not self.smoke_mode:
            raise WorkerError('SMOKE_MODE_REQUIRED', 'O primeiro treinamento real exige modo de smoke one-shot.')
        if not self.smoke_actor_profile_id or not self.smoke_training_run_id:
            raise WorkerError('SMOKE_SCOPE_MISSING', 'Ator e run autorizados não foram configurados no worker.')
        expiry = self.smoke_expiry()
        if not expiry or expiry <= datetime.now(timezone.utc):
            raise WorkerError('SMOKE_WINDOW_EXPIRED', 'A janela controlada do primeiro treinamento expirou.')
        if not all([self.r2_account_id, self.r2_access_key_id, self.r2_secret_access_key, self.r2_bucket_name]):
            raise WorkerError('R2_PRIVATE_CONFIG_MISSING', 'Credenciais privadas do R2 não configuradas.')
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.model_cache_root.mkdir(parents=True, exist_ok=True)
        self.smoke_lock_root.mkdir(parents=True, exist_ok=True)
