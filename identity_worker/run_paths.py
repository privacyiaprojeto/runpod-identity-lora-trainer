from __future__ import annotations

from pathlib import Path, PurePosixPath

from .errors import WorkerError


PERSISTENT_TRAINING_DIRNAME = "training-runs"
_REQUIRED_RUNTIME_PREFIX = PurePosixPath("/runpod-volume")


def _decode_mount_path(value: str) -> str:
    return (
        value.replace(r"\040", " ")
        .replace(r"\011", "\t")
        .replace(r"\012", "\n")
        .replace(r"\134", "\\")
    )


def _as_posix_contract_path(value: Path) -> PurePosixPath:
    """
    Validate the configured RunPod path lexically as POSIX.

    The worker runs on Linux, but the contract tests also run on Windows.
    Calling Path.resolve() on Windows rewrites /runpod-volume into
    C:/runpod-volume, which is a host-side artifact and must not change the
    Linux runtime contract being validated.
    """
    normalized = str(value).replace("\\", "/")
    path = PurePosixPath(normalized)
    if not path.is_absolute():
        raise WorkerError(
            "TRAINING_RUNTIME_ROOT_NOT_PERSISTENT",
            "O runtime do treinamento precisa usar um caminho POSIX absoluto sob /runpod-volume.",
            retryable=False,
        )
    return path


def _mount_points(mountinfo: str) -> tuple[PurePosixPath, ...]:
    points: list[PurePosixPath] = []
    for raw_line in mountinfo.splitlines():
        line = raw_line.strip()
        if not line or " - " not in line:
            continue
        left, _right = line.split(" - ", 1)
        fields = left.split()
        if len(fields) < 5:
            continue
        points.append(PurePosixPath(_decode_mount_path(fields[4])))
    return tuple(points)


def assert_persistent_runtime_root(
    runtime_root: Path,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> dict[str, str]:
    root_contract = _as_posix_contract_path(runtime_root)
    try:
        root_contract.relative_to(_REQUIRED_RUNTIME_PREFIX)
    except ValueError as exc:
        raise WorkerError(
            "TRAINING_RUNTIME_ROOT_NOT_PERSISTENT",
            "O runtime do treinamento precisa permanecer sob /runpod-volume.",
            retryable=False,
        ) from exc

    try:
        mountinfo = mountinfo_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise WorkerError(
            "TRAINING_RUNTIME_MOUNTINFO_UNAVAILABLE",
            "Não foi possível comprovar a montagem persistente do Network Volume.",
            retryable=True,
        ) from exc

    candidates = [
        point
        for point in _mount_points(mountinfo)
        if point == root_contract or point in root_contract.parents
    ]
    candidates.sort(key=lambda item: len(item.parts), reverse=True)

    if not candidates or candidates[0] == PurePosixPath("/"):
        raise WorkerError(
            "TRAINING_NETWORK_VOLUME_NOT_MOUNTED",
            "O caminho /runpod-volume existe, mas não foi comprovado como montagem persistente.",
            retryable=True,
        )

    mount_point = candidates[0]
    try:
        mount_point.relative_to(_REQUIRED_RUNTIME_PREFIX)
    except ValueError as exc:
        raise WorkerError(
            "TRAINING_NETWORK_VOLUME_NOT_MOUNTED",
            "O runtime não está apoiado no Network Volume esperado.",
            retryable=True,
        ) from exc

    # Filesystem operations intentionally use the native Path only after the
    # Linux mount contract has been proven from /proc/self/mountinfo.
    filesystem_root = runtime_root.resolve()
    filesystem_root.mkdir(parents=True, exist_ok=True)

    return {
        "runtime_root": str(filesystem_root),
        "mount_point": mount_point.as_posix(),
    }


def training_output_dir(
    runtime_root: Path,
    actor_profile_id: str,
    training_run_id: str,
) -> Path:
    root = runtime_root.resolve()
    output = (
        root
        / PERSISTENT_TRAINING_DIRNAME
        / actor_profile_id
        / training_run_id
        / "output"
    ).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise WorkerError(
            "TRAINING_OUTPUT_SCOPE_INVALID",
            "O diretório de checkpoints saiu do runtime persistente autorizado.",
            retryable=False,
        ) from exc
    return output


def prepare_training_output_dir(
    runtime_root: Path,
    actor_profile_id: str,
    training_run_id: str,
) -> Path:
    output = training_output_dir(runtime_root, actor_profile_id, training_run_id)
    if output.exists():
        existing = tuple(output.iterdir())
        if existing:
            raise WorkerError(
                "TRAINING_OUTPUT_ALREADY_EXISTS",
                "O diretório persistente deste run já contém artefatos; mistura ou sobrescrita foi bloqueada.",
                retryable=False,
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output
