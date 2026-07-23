from __future__ import annotations
from pathlib import Path
from .errors import WorkerError
from .hashing import sha256_file

def client(settings):
    # O SDK de storage é obrigatório no container, mas não deve ser carregado
    # durante a coleta dos testes de contrato, que instalam apenas pytest.
    import boto3

    return boto3.client(
        's3',
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name='auto',
    )

def download_private(s3, bucket: str, key: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket, key, str(destination))
    except Exception as exc:
        raise WorkerError('R2_DOWNLOAD_FAILED', f'Falha ao baixar material privado: {key}', retryable=True) from exc
    if sha256_file(destination) != expected_sha256.lower():
        raise WorkerError('ASSET_CHECKSUM_MISMATCH', f'Checksum divergente: {key}')
    return destination

def upload_private(s3, path: Path, bucket: str, key: str, metadata: dict[str, str]) -> dict:
    try:
        s3.upload_file(str(path), bucket, key, ExtraArgs={'ContentType':'application/octet-stream','CacheControl':'private, no-store','Metadata':metadata})
    except Exception as exc:
        raise WorkerError('R2_UPLOAD_FAILED', 'Falha ao enviar adapter privado.', retryable=True) from exc
    return {'r2_bucket':bucket,'r2_key':key,'sha256':sha256_file(path),'byte_size':path.stat().st_size}
