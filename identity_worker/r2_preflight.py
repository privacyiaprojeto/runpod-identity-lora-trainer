from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .contracts import R2PreflightObject
from .errors import WorkerError


def _error_code(error: Exception) -> str:
    response = getattr(error, 'response', None)
    if isinstance(response, dict):
        details = response.get('Error')
        if isinstance(details, dict):
            return str(details.get('Code') or '').strip()
    return ''


def _key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]


def _masked_bucket(bucket: str) -> str:
    value = str(bucket or '').strip()
    if len(value) <= 4:
        return '*' * len(value)
    return f'{value[:3]}***{value[-2:]}'


def _metadata_sha256(metadata: dict[str, Any]) -> str:
    normalized = {str(key).strip().lower(): str(value).strip().lower() for key, value in metadata.items()}
    for name in ('sha256', 'checksum-sha256', 'content-sha256', 'asset-sha256'):
        value = normalized.get(name, '')
        if value:
            return value
    return ''


def probe_private_r2_metadata(
    s3,
    bucket: str,
    objects: Iterable[R2PreflightObject],
) -> dict[str, Any]:
    """Valida acesso privado por HEAD, sem listar, baixar, escrever ou apagar objetos."""

    unique: dict[tuple[str, str], R2PreflightObject] = {}
    for item in objects:
        unique.setdefault((item.bucket, item.key), item)

    if not unique:
        raise WorkerError('R2_PREFLIGHT_OBJECTS_REQUIRED', 'O preflight exige ao menos um objeto privado.')

    total_bytes = 0
    metadata_checksum_present = 0
    metadata_checksum_verified = 0
    checked: list[dict[str, Any]] = []

    for (item_bucket, key), item in unique.items():
        if item_bucket != bucket:
            raise WorkerError('R2_PREFLIGHT_BUCKET_SCOPE_MISMATCH', 'Objeto fora do bucket privado autorizado.')

        try:
            response = s3.head_object(Bucket=bucket, Key=key)
        except Exception as error:
            code = _error_code(error)
            suffix = f' ({code})' if code else ''
            raise WorkerError(
                'R2_PREFLIGHT_HEAD_FAILED',
                f'Falha ao consultar metadados do objeto privado {item.sample_id}/{item.role}{suffix}.',
                retryable=code in {'429', '500', '502', '503', '504', 'InternalError', 'SlowDown', 'ServiceUnavailable'},
            ) from error

        byte_size = int(response.get('ContentLength') or 0)
        if byte_size <= 0:
            raise WorkerError(
                'R2_PREFLIGHT_EMPTY_OBJECT',
                f'Objeto privado vazio ou sem tamanho válido: {item.sample_id}/{item.role}.',
            )

        metadata = response.get('Metadata') or {}
        stored_sha256 = _metadata_sha256(metadata if isinstance(metadata, dict) else {})
        checksum_state = 'metadata_absent'
        if stored_sha256:
            metadata_checksum_present += 1
            if stored_sha256 != item.expected_sha256.lower():
                raise WorkerError(
                    'R2_PREFLIGHT_METADATA_CHECKSUM_MISMATCH',
                    f'Checksum de metadados divergente: {item.sample_id}/{item.role}.',
                )
            metadata_checksum_verified += 1
            checksum_state = 'metadata_verified'

        total_bytes += byte_size
        checked.append({
            'sample_id': item.sample_id,
            'role': item.role,
            'key_fingerprint': _key_fingerprint(key),
            'byte_size': byte_size,
            'content_type': str(response.get('ContentType') or ''),
            'checksum_state': checksum_state,
        })

    return {
        'bucket_masked': _masked_bucket(bucket),
        'bucket_fingerprint': hashlib.sha256(bucket.encode('utf-8')).hexdigest()[:12],
        'objects_checked': len(checked),
        'total_bytes': total_bytes,
        'metadata_checksum_present': metadata_checksum_present,
        'metadata_checksum_verified': metadata_checksum_verified,
        'objects': checked,
    }
