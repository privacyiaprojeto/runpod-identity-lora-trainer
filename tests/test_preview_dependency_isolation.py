import builtins
import importlib
import sys


def test_preview_contract_imports_without_gpu_or_storage_sdk(monkeypatch):
    blocked={'boto3','torch','diffsynth','PIL'}
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in blocked:
            raise AssertionError(f'eager dependency import forbidden: {name}')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    for module in ['identity_worker.preview','identity_worker.contracts','identity_worker.config']:
        sys.modules.pop(module,None)
        importlib.import_module(module)
