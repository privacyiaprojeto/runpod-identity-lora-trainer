from __future__ import annotations
import csv
import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from .contracts import (
    H2_CONTRACT_VERSION,
    H2_HEIGHT,
    H2_WIDTH,
)
from .errors import WorkerError
from .hashing import sha256_file
from .storage import download_private

def _materialize_legacy_dataset(request, settings, work_dir: Path, s3) -> tuple[Path, Path]:
    dataset_root = work_dir / 'dataset'
    assets_root = dataset_root / 'assets'
    rows = []
    cache: dict[tuple[str,str,str], Path] = {}
    for sample in request.payload['dataset']['samples']:
        video_ref = sample['video_source']; image_ref = sample['reference_image_source']
        vkey=(video_ref['bucket'],video_ref['key'],sample['video_sha256'])
        ikey=(image_ref['bucket'],image_ref['key'],sample['reference_image_sha256'])
        if vkey not in cache:
            cache[vkey]=download_private(s3,*vkey[:2],assets_root/f"video_{len(cache):03d}.mp4",vkey[2])
        if ikey not in cache:
            cache[ikey]=download_private(s3,*ikey[:2],assets_root/f"image_{len(cache):03d}.jpg",ikey[2])
        video_rel=cache[vkey].relative_to(dataset_root).as_posix(); image_rel=cache[ikey].relative_to(dataset_root).as_posix()
        rows.append({'video':video_rel,'vace_video':video_rel,'vace_reference_image':image_rel,'prompt':sample['prompt']})
    metadata_path=dataset_root/'metadata.csv'
    dataset_root.mkdir(parents=True,exist_ok=True)
    with metadata_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['video','vace_video','vace_reference_image','prompt']); writer.writeheader(); writer.writerows(rows)
    return dataset_root, metadata_path

# ============================================================
# POD7C_H2_DATASET_V1
#
# H1/V2 remains above as historical behavior.
# H2/V3:
#   video      = normalized RGB target
#   vace_video = aligned appearance-reduced softedge
# ============================================================

H2_DATASET_SCHEMA = (
    "privacy-identity-compiled-training-manifest-v1"
)

H2_RECIPE_VERSION = (
    "pod7c-h2-portrait-aligned-softedge-v1"
)

H2_TARGET_NORMALIZATION = (
    "rgb_aspect_preserving_center_crop_v1"
)

H2_CONTROL_REPRESENTATION = (
    "softedge_ffmpeg_edgedetect_v1"
)

H2_FPS = 16
H2_FRAMES = 17

H2_CLIP_SECONDS = (
    H2_FRAMES /
    H2_FPS
)

H2_SOFTEDGE_FILTER = (
    "format=gray,"
    "edgedetect="
    "low=0.0784314:"
    "high=0.196078:"
    "mode=wires,"
    "format=yuv420p"
)


def _run_h2_media(
    command: list[str],
    code: str,
    message: str,
) -> subprocess.CompletedProcess[str]:

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    except FileNotFoundError as exc:
        raise WorkerError(
            code,
            message,
            retryable=False,
        ) from exc

    if result.returncode != 0:

        detail = (
            result.stderr or
            result.stdout or
            ""
        ).strip()[-1500:]

        raise WorkerError(
            code,
            (
                f"{message} "
                f"{detail}"
            ).strip(),
            retryable=False,
        )

    return result


def _h2_fps_value(
    value: str,
) -> float:

    if (
        not value or
        value == "0/0"
    ):
        return 0.0

    try:
        return float(
            Fraction(value)
        )

    except (
        ValueError,
        ZeroDivisionError,
    ):
        return 0.0


def _probe_h2_video(
    path: Path,
) -> dict:

    result = _run_h2_media(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream="
                "width,height,"
                "avg_frame_rate,"
                "nb_read_frames"
            ),
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        "H2_FFPROBE_FAILED",
        "Falha auditando video H2.",
    )

    try:
        payload = json.loads(
            result.stdout or "{}"
        )

        stream = (
            payload.get(
                "streams"
            ) or [{}]
        )[0]

        rate = str(
            stream.get(
                "avg_frame_rate"
            ) or ""
        )

        return {
            "width":
                int(
                    stream.get(
                        "width"
                    ) or 0
                ),

            "height":
                int(
                    stream.get(
                        "height"
                    ) or 0
                ),

            "avg_frame_rate":
                rate,

            "fps":
                _h2_fps_value(
                    rate
                ),

            "frame_count":
                int(
                    stream.get(
                        "nb_read_frames"
                    ) or 0
                ),

            "duration_seconds":
                float(
                    (
                        payload.get(
                            "format"
                        ) or {}
                    ).get(
                        "duration"
                    ) or 0.0
                ),
        }

    except Exception as exc:
        raise WorkerError(
            "H2_FFPROBE_INVALID",
            "Metadados H2 invalidos.",
            retryable=False,
        ) from exc


def _assert_h2_media_runtime() -> str:

    filters = _run_h2_media(
        [
            "ffmpeg",
            "-hide_banner",
            "-filters",
        ],
        "H2_FFMPEG_MISSING",
        "FFmpeg ausente no runtime H2.",
    )

    if (
        "edgedetect"
        not in (
            filters.stdout or ""
        )
    ):
        raise WorkerError(
            "H2_EDGEDETECT_MISSING",
            (
                "Filtro edgedetect "
                "ausente no runtime H2."
            ),
            retryable=False,
        )

    version = _run_h2_media(
        [
            "ffmpeg",
            "-version",
        ],
        "H2_FFMPEG_VERSION_FAILED",
        "Falha lendo versao FFmpeg.",
    )

    return (
        version.stdout or ""
    ).splitlines()[0].strip()


def _h2_clip_start_seconds(
    source_probe: dict,
    occurrence_index: int,
    occurrence_count: int,
) -> float:

    duration = float(
        source_probe.get(
            "duration_seconds"
        ) or 0.0
    )

    max_start = max(
        0.0,
        duration -
        H2_CLIP_SECONDS -
        0.10,
    )

    if max_start <= 0.0:
        return 0.0

    fraction = (
        occurrence_index + 1
    ) / (
        occurrence_count + 1
    )

    return round(
        max_start * fraction,
        6,
    )


def _normalize_h2_target(
    source: Path,
    destination: Path,
    start_seconds: float,
) -> dict:

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_filter = (
        f"fps={H2_FPS},"
        f"scale="
        f"{H2_WIDTH}:{H2_HEIGHT}:"
        "force_original_aspect_ratio="
        "increase:"
        "flags=lanczos,"
        f"crop={H2_WIDTH}:{H2_HEIGHT},"
        "format=yuv420p"
    )

    _run_h2_media(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",

            "-ss",
            f"{start_seconds:.6f}",

            "-i",
            str(source),

            "-map",
            "0:v:0",

            "-an",
            "-sn",
            "-dn",

            "-vf",
            video_filter,

            "-frames:v",
            str(H2_FRAMES),

            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "18",

            "-pix_fmt",
            "yuv420p",

            "-threads",
            "1",

            "-movflags",
            "+faststart",

            str(destination),
        ],
        "H2_TARGET_NORMALIZATION_FAILED",
        "Falha normalizando target H2.",
    )

    probe = _probe_h2_video(
        destination
    )

    if (
        probe["width"] !=
        H2_WIDTH

        or

        probe["height"] !=
        H2_HEIGHT

        or

        probe["frame_count"] !=
        H2_FRAMES

        or

        abs(
            probe["fps"] -
            H2_FPS
        ) > 0.001
    ):
        raise WorkerError(
            "H2_TARGET_GEOMETRY_INVALID",
            (
                "Target H2 fora do "
                f"contrato: {probe}"
            ),
            retryable=False,
        )

    return probe


def _derive_h2_softedge(
    target: Path,
    destination: Path,
) -> dict:

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _run_h2_media(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",

            "-i",
            str(target),

            "-map",
            "0:v:0",

            "-an",
            "-sn",
            "-dn",

            "-vf",
            H2_SOFTEDGE_FILTER,

            "-frames:v",
            str(H2_FRAMES),

            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "18",

            "-pix_fmt",
            "yuv420p",

            "-threads",
            "1",

            "-movflags",
            "+faststart",

            str(destination),
        ],
        "H2_STRUCTURAL_CONTROL_FAILED",
        "Falha gerando softedge H2.",
    )

    target_sha = sha256_file(
        target
    )

    control_sha = sha256_file(
        destination
    )

    if target_sha == control_sha:
        raise WorkerError(
            "H2_RAW_RGB_SELF_CONDITIONING_BLOCKED",
            (
                "Target RGB e vace_video "
                "ficaram identicos."
            ),
            retryable=False,
        )

    target_probe = _probe_h2_video(
        target
    )

    control_probe = _probe_h2_video(
        destination
    )

    if (
        target_probe["width"] !=
        control_probe["width"]

        or

        target_probe["height"] !=
        control_probe["height"]

        or

        target_probe["frame_count"] !=
        control_probe["frame_count"]

        or

        abs(
            target_probe["fps"] -
            control_probe["fps"]
        ) > 0.001
    ):
        raise WorkerError(
            "H2_TARGET_CONTROL_ALIGNMENT_FAILED",
            (
                "Target RGB e softedge "
                "nao estao alinhados."
            ),
            retryable=False,
        )

    return {
        "sha256":
            control_sha,

        "probe":
            control_probe,
    }


def _materialize_h2_dataset(
    request,
    settings,
    work_dir: Path,
    s3,
) -> tuple[Path, Path]:

    del settings

    trigger_token = str(
        request.payload.get(
            "trigger_token"
        ) or ""
    ).strip()

    samples = list(
        request.payload[
            "dataset"
        ][
            "samples"
        ]
    )

    ffmpeg_version = (
        _assert_h2_media_runtime()
    )

    dataset_root = (
        work_dir /
        "dataset"
    )

    assets_root = (
        dataset_root /
        "assets"
    )

    targets_root = (
        dataset_root /
        "targets"
    )

    controls_root = (
        dataset_root /
        "controls"
    )

    assets_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    controls_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache: dict[
        tuple[str, str, str],
        Path,
    ] = {}

    source_probes: dict[
        str,
        dict,
    ] = {}

    occurrence_total: dict[
        str,
        int,
    ] = {}

    occurrence_seen: dict[
        str,
        int,
    ] = {}

    for sample in samples:

        video_sha = str(
            sample[
                "video_sha256"
            ]
        ).lower()

        occurrence_total[
            video_sha
        ] = (
            occurrence_total.get(
                video_sha,
                0,
            ) +
            1
        )

    rows = []
    compiled_samples = []

    for (
        index,
        sample,
    ) in enumerate(
        samples,
        start=1,
    ):

        prompt = str(
            sample.get(
                "prompt"
            ) or ""
        ).strip()

        if (
            not prompt or
            trigger_token not in prompt
        ):
            raise WorkerError(
                "H2_PROMPT_TRIGGER_MISSING",
                (
                    "Prompt H2 perdeu "
                    "o trigger token."
                ),
                retryable=False,
            )

        video_ref = (
            sample[
                "video_source"
            ]
        )

        image_ref = (
            sample[
                "reference_image_source"
            ]
        )

        video_sha = str(
            sample[
                "video_sha256"
            ]
        ).lower()

        image_sha = str(
            sample[
                "reference_image_sha256"
            ]
        ).lower()

        vkey = (
            video_ref["bucket"],
            video_ref["key"],
            video_sha,
        )

        ikey = (
            image_ref["bucket"],
            image_ref["key"],
            image_sha,
        )

        if vkey not in cache:

            cache[vkey] = (
                download_private(
                    s3,
                    *vkey[:2],
                    (
                        assets_root /
                        (
                            f"source_"
                            f"{video_sha}"
                            ".mp4"
                        )
                    ),
                    vkey[2],
                )
            )

        if ikey not in cache:

            suffix = (
                Path(
                    image_ref["key"]
                ).suffix.lower()
                or
                ".jpg"
            )

            cache[ikey] = (
                download_private(
                    s3,
                    *ikey[:2],
                    (
                        assets_root /
                        (
                            f"reference_"
                            f"{image_sha}"
                            f"{suffix}"
                        )
                    ),
                    ikey[2],
                )
            )

        if (
            video_sha
            not in source_probes
        ):
            source_probes[
                video_sha
            ] = _probe_h2_video(
                cache[vkey]
            )

        occurrence_index = (
            occurrence_seen.get(
                video_sha,
                0,
            )
        )

        start_seconds = (
            _h2_clip_start_seconds(
                source_probes[
                    video_sha
                ],
                occurrence_index,
                occurrence_total[
                    video_sha
                ],
            )
        )

        occurrence_seen[
            video_sha
        ] = (
            occurrence_index +
            1
        )

        sample_id = str(
            sample.get(
                "sample_id"
            ) or
            f"sample-{index:03d}"
        )

        safe_id = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            sample_id,
        )[:64]

        if not safe_id:
            safe_id = (
                f"sample-{index:03d}"
            )

        target_path = (
            targets_root /
            (
                f"{index:03d}_"
                f"{safe_id}_"
                "target.mp4"
            )
        )

        control_path = (
            controls_root /
            (
                f"{index:03d}_"
                f"{safe_id}_"
                "softedge.mp4"
            )
        )

        target_probe = (
            _normalize_h2_target(
                cache[vkey],
                target_path,
                start_seconds,
            )
        )

        control_info = (
            _derive_h2_softedge(
                target_path,
                control_path,
            )
        )

        target_sha = (
            sha256_file(
                target_path
            )
        )

        target_rel = (
            target_path
            .relative_to(
                dataset_root
            )
            .as_posix()
        )

        control_rel = (
            control_path
            .relative_to(
                dataset_root
            )
            .as_posix()
        )

        image_rel = (
            cache[ikey]
            .relative_to(
                dataset_root
            )
            .as_posix()
        )

        if target_rel == control_rel:
            raise WorkerError(
                "H2_RAW_RGB_SELF_CONDITIONING_BLOCKED",
                (
                    "video e vace_video "
                    "resolveram para "
                    "o mesmo arquivo."
                ),
                retryable=False,
            )

        rows.append(
            {
                "video":
                    target_rel,

                "vace_video":
                    control_rel,

                "vace_reference_image":
                    image_rel,

                "prompt":
                    prompt,
            }
        )

        compiled_samples.append(
            {
                "sample_index":
                    index,

                "sample_id":
                    sample_id,

                "prompt":
                    prompt,

                "trigger_token":
                    trigger_token,

                "source_video_sha256":
                    video_sha,

                "source_video_probe":
                    source_probes[
                        video_sha
                    ],

                "clip_start_seconds":
                    start_seconds,

                "target": {
                    "path":
                        target_rel,

                    "sha256":
                        target_sha,

                    "probe":
                        target_probe,

                    "raw_rgb":
                        True,
                },

                "control": {
                    "path":
                        control_rel,

                    "sha256":
                        control_info[
                            "sha256"
                        ],

                    "representation":
                        H2_CONTROL_REPRESENTATION,

                    "raw_rgb":
                        False,

                    "appearance_reduced":
                        True,

                    "probe":
                        control_info[
                            "probe"
                        ],
                },

                "reference_image": {
                    "path":
                        image_rel,

                    "sha256":
                        image_sha,
                },
            }
        )

    metadata_path = (
        dataset_root /
        "metadata.csv"
    )

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video",
                "vace_video",
                "vace_reference_image",
                "prompt",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    manifest = {
        "schema_version":
            H2_DATASET_SCHEMA,

        "recipe_version":
            H2_RECIPE_VERSION,

        "actor_profile_id":
            request.actor_profile_id,

        "training_run_id":
            request.training_run_id,

        "dataset_manifest_sha256":
            request.payload[
                "dataset_manifest_sha256"
            ],

        "trigger_token":
            trigger_token,

        "training_geometry": {
            "width":
                H2_WIDTH,

            "height":
                H2_HEIGHT,

            "fps":
                H2_FPS,

            "frames":
                H2_FRAMES,
        },

        "target_normalization":
            H2_TARGET_NORMALIZATION,

        "control_representation":
            H2_CONTROL_REPRESENTATION,

        "raw_rgb_self_conditioning_forbidden":
            True,

        "appearance_reduced_control_required":
            True,

        "ffmpeg_version":
            ffmpeg_version,

        "sample_count":
            len(
                compiled_samples
            ),

        "samples":
            compiled_samples,
    }

    manifest_path = (
        dataset_root /
        "compiled_training_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest_sha = (
        sha256_file(
            manifest_path
        )
    )

    (
        dataset_root /
        "compiled_training_manifest.sha256"
    ).write_text(
        manifest_sha + "\n",
        encoding="ascii",
    )

    return (
        dataset_root,
        metadata_path,
    )


def materialize_dataset(
    request,
    settings,
    work_dir: Path,
    s3,
) -> tuple[Path, Path]:

    if (
        request.payload.get(
            "contract_version"
        ) ==
        H2_CONTRACT_VERSION
    ):
        return (
            _materialize_h2_dataset(
                request,
                settings,
                work_dir,
                s3,
            )
        )

    return (
        _materialize_legacy_dataset(
            request,
            settings,
            work_dir,
            s3,
        )
    )
