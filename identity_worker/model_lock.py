from __future__ import annotations
from pathlib import Path
from huggingface_hub import hf_hub_download
from .errors import WorkerError
from .storage import sha256_file

def materialize_model(request, settings) -> list[str]:
    model=request.payload['model']; repo=model['repository']; revision=model['revision']
    paths=[]
    for artifact in model['artifacts']:
        path=Path(hf_hub_download(repo_id=repo,filename=artifact['path'],revision=revision,token=settings.hf_token or None,cache_dir=str(settings.model_cache_root)))
        if sha256_file(path) != artifact['sha256'].lower():
            raise WorkerError('MODEL_ARTIFACT_CHECKSUM_MISMATCH', f"Artefato divergente: {artifact['path']}")
        paths.append(str(path))
    return paths
