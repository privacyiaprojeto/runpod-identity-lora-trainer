from pathlib import Path

from identity_worker.dataset import (
    H2_CONTROL_REPRESENTATION,
    H2_FPS,
    H2_FRAMES,
    H2_RECIPE_VERSION,
    _h2_clip_start_seconds,
)


def test_h2_dataset_contract_constants():

    assert (
        H2_RECIPE_VERSION ==
        "pod7c-h2-portrait-aligned-softedge-v1"
    )

    assert (
        H2_CONTROL_REPRESENTATION ==
        "softedge_ffmpeg_edgedetect_v1"
    )

    assert H2_FPS == 16
    assert H2_FRAMES == 17


def test_repeated_video_samples_get_distinct_windows():

    probe = {
        "duration_seconds": 30.0,
    }

    starts = [
        _h2_clip_start_seconds(
            probe,
            index,
            3,
        )
        for index in range(3)
    ]

    assert starts[0] > 0
    assert starts[0] < starts[1] < starts[2]


def test_short_video_falls_back_to_start():

    probe = {
        "duration_seconds": 0.5,
    }

    assert (
        _h2_clip_start_seconds(
            probe,
            0,
            1,
        ) ==
        0.0
    )


def test_h2_source_separates_rgb_target_and_vace_control():

    source = (
        Path(__file__)
        .parents[1]
        .joinpath(
            "identity_worker",
            "dataset.py",
        )
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "POD7C_H2_DATASET_V1"
        in source
    )

    assert (
        "H2_RAW_RGB_SELF_CONDITIONING_BLOCKED"
        in source
    )

    assert (
        "compiled_training_manifest.json"
        in source
    )

    h2 = (
        source
        .split(
            "def _materialize_h2_dataset(",
            1,
        )[1]
        .split(
            "def materialize_dataset(",
            1,
        )[0]
    )

    assert (
        '"vace_video":'
        in h2
    )

    assert (
        "control_rel"
        in h2
    )

    assert (
        '"vace_video":video_rel'
        not in h2.replace(
            " ",
            ""
        )
    )

    assert (
        '"raw_rgb":'
        in h2
    )

    assert (
        '"appearance_reduced":'
        in h2
    )


def test_legacy_h1_is_still_present_separately():

    source = (
        Path(__file__)
        .parents[1]
        .joinpath(
            "identity_worker",
            "dataset.py",
        )
        .read_text(
            encoding="utf-8"
        )
    )

    legacy = (
        source
        .split(
            "def _materialize_legacy_dataset(",
            1,
        )[1]
        .split(
            "# POD7C_H2_DATASET_V1",
            1,
        )[0]
    )

    assert (
        "'vace_video':video_rel"
        in legacy.replace(
            " ",
            ""
        )
    )
