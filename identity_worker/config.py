from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from .errors import WorkerError

def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() == 'true'

@dataclass(frozen=True)
class Settings:
    allow_training: bool = _bool('PRIVACY_LORA_ALLOW_TRAINING', False)
    dry_run_only: bool = _bool('PRIVACY_LORA_DRY_RUN_ONLY', True)
    r2_account_id: str = os.getenv('R2_ACCOUNT_ID', '').strip()
    r2_access_key_id: str = os.getenv('R2_ACCESS_KEY_ID', '').strip()
    r2_secret_access_key: str = os.getenv('R2_SECRET_ACCESS_KEY', '').strip()
    r2_bucket_name: str = os.getenv('R2_BUCKET_NAME', '').strip()
    hf_token: str = os.getenv('HF_TOKEN', '').strip()
    app_root: Path = Path(os.getenv('APP_ROOT', '/app'))
    runtime_root: Path = Path(os.getenv('RUNTIME_ROOT', '/runpod-volume/privacy-identity-lora'))
    model_cache_root: Path = Path(os.getenv('MODEL_CACHE_ROOT', '/runpod-volume/models/identity-lora'))
    diffsynth_root: Path = Path(os.getenv('DIFFSYNTH_ROOT', '/opt/DiffSynth-Studio'))

    @property
    def r2_endpoint_url(self) -> str:
        return f'https://{self.r2_account_id}.r2.cloudflarestorage.com'

    def validate_runtime(self) -> None:
        if self.dry_run_only or not self.allow_training:
            raise WorkerError('TRAINING_DISABLED', 'O treinamento real permanece desligado por política.')
        if not all([self.r2_account_id, self.r2_access_key_id, self.r2_secret_access_key, self.r2_bucket_name]):
            raise WorkerError('R2_PRIVATE_CONFIG_MISSING', 'Credenciais privadas do R2 não configuradas.')
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.model_cache_root.mkdir(parents=True, exist_ok=True)
