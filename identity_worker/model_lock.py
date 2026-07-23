from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import WorkerError
from .hashing import sha256_file

_EXPECTED_REPOSITORY = "Wan-AI/Wan2.1-VACE-14B"
_SHARD_RE = re.compile(r"^diffusion_pytorch_model-(\d{5})-of-00007\.safetensors$")
_TEXT_ENCODER_NAME = "models_t5_umt5-xxl-enc-bf16.pth"
_VAE_NAME = "Wan2.1_VAE.pth"


def hf_hub_download(**kwargs):
    """Importa o cliente HF apenas quando a materialização real é executada.

    O preflight/testes locais não devem exigir que o ambiente de desenvolvimento
    tenha todas as dependências do container instaladas. No worker Docker, a
    versão permanece fixada em requirements.txt e validada pelo runtime preflight.
    """
    from huggingface_hub import hf_hub_download as _hf_hub_download

    return _hf_hub_download(**kwargs)


@dataclass(frozen=True)
class MaterializedModelBinding:
    repository: str
    revision: str
    diffusion_shards: tuple[str, ...]
    text_encoder_path: str
    vae_path: str

    def diffsynth_model_paths(self) -> list[object]:
        # DiffSynth accepts a list of model configs. The first config must be
        # a single grouped list containing every shard of the VACE checkpoint.
        return [list(self.diffusion_shards), self.text_encoder_path, self.vae_path]


def _ensure_in_cache(path: Path, cache_root: Path) -> None:
    try:
        path.resolve().relative_to(cache_root.resolve())
    except ValueError as exc:
        raise WorkerError(
            "MODEL_CACHE_SCOPE_MISMATCH",
            "O artefato do modelo-base está fora do cache privado autorizado.",
            retryable=False,
        ) from exc


def _download_locked_artifact(repo: str, revision: str, artifact_path: str, settings) -> Path:
    try:
        resolved = Path(
            hf_hub_download(
                repo_id=repo,
                filename=artifact_path,
                revision=revision,
                token=settings.hf_token or None,
                cache_dir=str(settings.model_cache_root),
                local_files_only=True,
            )
        )
    except Exception as exc:
        raise WorkerError(
            "MODEL_CACHE_MISS",
            "O modelo-base congelado não está completo no cache privado do endpoint.",
            retryable=True,
        ) from exc
    _ensure_in_cache(resolved, Path(settings.model_cache_root))
    return resolved


def materialize_model(request, settings) -> MaterializedModelBinding:
    model = request.payload["model"]
    repository = str(model.get("repository") or "").strip()
    revision = str(model.get("revision") or "").strip()
    artifacts = list(model.get("artifacts") or [])

    if repository != _EXPECTED_REPOSITORY:
        raise WorkerError(
            "MODEL_BINDING_REPOSITORY_MISMATCH",
            "O run não está vinculado ao repositório homologado do VACE-14B.",
            retryable=False,
        )
    if not revision or len(artifacts) != 9:
        raise WorkerError(
            "MODEL_BINDING_INVALID",
            "O lock do modelo-base não possui revisão e nove artefatos válidos.",
            retryable=False,
        )

    resolved_by_name: dict[str, str] = {}
    for artifact in artifacts:
        artifact_path = str(artifact.get("path") or "").strip()
        expected_sha = str(artifact.get("sha256") or "").strip().lower()
        if not artifact_path or len(expected_sha) != 64:
            raise WorkerError("MODEL_BINDING_INVALID", "Artefato inválido no lock do modelo-base.")
        resolved = _download_locked_artifact(repository, revision, artifact_path, settings)
        if sha256_file(resolved) != expected_sha:
            raise WorkerError(
                "MODEL_ARTIFACT_CHECKSUM_MISMATCH",
                f"Artefato divergente: {artifact_path}",
                retryable=False,
            )
        name = Path(artifact_path).name
        if name in resolved_by_name:
            raise WorkerError("MODEL_BINDING_INVALID", f"Artefato duplicado no lock: {name}")
        resolved_by_name[name] = str(resolved)

    shard_items: list[tuple[int, str]] = []
    for name, path in resolved_by_name.items():
        match = _SHARD_RE.match(name)
        if match:
            shard_items.append((int(match.group(1)), path))
    shard_items.sort(key=lambda item: item[0])
    if [index for index, _ in shard_items] != list(range(1, 8)):
        raise WorkerError(
            "MODEL_BINDING_SHARDS_INVALID",
            "O checkpoint VACE precisa conter exatamente os sete shards oficiais em sequência.",
            retryable=False,
        )

    text_encoder_path = resolved_by_name.get(_TEXT_ENCODER_NAME)
    vae_path = resolved_by_name.get(_VAE_NAME)
    if not text_encoder_path or not vae_path:
        raise WorkerError(
            "MODEL_BINDING_COMPONENT_MISSING",
            "O modelo-base congelado não contém o text encoder e o VAE homologados.",
            retryable=False,
        )

    recognized_names = {name for name in resolved_by_name if _SHARD_RE.match(name)} | {
        _TEXT_ENCODER_NAME,
        _VAE_NAME,
    }
    if recognized_names != set(resolved_by_name):
        unexpected = sorted(set(resolved_by_name) - recognized_names)
        raise WorkerError(
            "MODEL_BINDING_UNEXPECTED_ARTIFACT",
            f"O lock contém artefato não homologado: {unexpected[0]}",
            retryable=False,
        )

    return MaterializedModelBinding(
        repository=repository,
        revision=revision,
        diffusion_shards=tuple(path for _, path in shard_items),
        text_encoder_path=text_encoder_path,
        vae_path=vae_path,
    )
