from __future__ import annotations

import builtins
import hashlib
import importlib
import sys
from pathlib import Path


def test_contract_modules_import_without_runtime_cloud_sdks(monkeypatch):
    """Os contract-tests do GitHub Actions instalam apenas pytest.

    Importar os módulos puros do binding/preflight não pode carregar boto3 nem
    huggingface_hub; esses SDKs só são necessários quando o worker executa I/O real.
    """
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split('.', 1)[0]
        if root in {'boto3', 'huggingface_hub'}:
            raise ModuleNotFoundError(f'blocked runtime dependency: {root}')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    for module_name in (
        'identity_worker.trainer',
        'identity_worker.model_lock',
        'identity_worker.storage',
        'identity_worker.hashing',
    ):
        sys.modules.pop(module_name, None)

    importlib.import_module('identity_worker.model_lock')
    importlib.import_module('identity_worker.trainer')
    importlib.import_module('identity_worker.storage')


def test_dependency_free_sha256_helper(tmp_path: Path):
    payload = b'privacy-ia-model-binding-contract'
    path = tmp_path / 'artifact.bin'
    path.write_bytes(payload)

    from identity_worker.hashing import sha256_file

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
