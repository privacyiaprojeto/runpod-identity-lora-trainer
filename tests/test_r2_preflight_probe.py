from __future__ import annotations

import pytest

from identity_worker.contracts import R2PreflightObject
from identity_worker.errors import WorkerError
from identity_worker.r2_preflight import probe_private_r2_metadata


class FakeS3:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def head_object(self, *, Bucket, Key):
        self.calls.append(('head_object', Bucket, Key))
        value = self.responses[(Bucket, Key)]
        if isinstance(value, Exception):
            raise value
        return value

    def __getattr__(self, name):
        if name in {'get_object', 'download_file', 'upload_file', 'put_object', 'delete_object', 'list_objects_v2'}:
            raise AssertionError(f'forbidden S3 operation: {name}')
        raise AttributeError(name)


def item(key, sha='a' * 64, role='video_source'):
    return R2PreflightObject('sample-01', role, 'privacy-media', key, sha)


def test_probe_uses_head_only_and_masks_storage_identifiers():
    first = 'vault/actor/sample/video.mp4'
    second = 'vault/actor/sample/image.jpg'
    s3 = FakeS3({
        ('privacy-media', first): {'ContentLength': 100, 'ContentType': 'video/mp4', 'Metadata': {}},
        ('privacy-media', second): {'ContentLength': 50, 'ContentType': 'image/jpeg', 'Metadata': {'sha256': 'b' * 64}},
    })

    result = probe_private_r2_metadata(
        s3,
        'privacy-media',
        [item(first), item(second, 'b' * 64, 'reference_image_source')],
    )

    assert result['bucket_masked'] == 'pri***ia'
    assert result['objects_checked'] == 2
    assert result['total_bytes'] == 150
    assert result['metadata_checksum_present'] == 1
    assert result['metadata_checksum_verified'] == 1
    assert all(call[0] == 'head_object' for call in s3.calls)
    assert first not in str(result)
    assert second not in str(result)


def test_probe_rejects_empty_object():
    key = 'vault/actor/sample/video.mp4'
    s3 = FakeS3({('privacy-media', key): {'ContentLength': 0, 'Metadata': {}}})
    with pytest.raises(WorkerError) as error:
        probe_private_r2_metadata(s3, 'privacy-media', [item(key)])
    assert error.value.code == 'R2_PREFLIGHT_EMPTY_OBJECT'


def test_probe_rejects_metadata_checksum_mismatch():
    key = 'vault/actor/sample/video.mp4'
    s3 = FakeS3({
        ('privacy-media', key): {'ContentLength': 100, 'Metadata': {'sha256': 'b' * 64}},
    })
    with pytest.raises(WorkerError) as error:
        probe_private_r2_metadata(s3, 'privacy-media', [item(key)])
    assert error.value.code == 'R2_PREFLIGHT_METADATA_CHECKSUM_MISMATCH'
