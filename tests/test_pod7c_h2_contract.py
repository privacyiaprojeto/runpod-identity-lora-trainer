from pathlib import Path
import runpy

import pytest

from identity_worker.contracts import (
    CONTRACT_VERSION,
    DIT_TRAINING_PROFILE,
    H2_CONTRACT_VERSION,
    H2_HEIGHT,
    H2_TRAINING_PROFILE,
    H2_WIDTH,
    parse_training_request,
)

from identity_worker.errors import WorkerError


_CANONICAL = runpy.run_path(
    str(
        Path(__file__).with_name(
            "test_contracts.py"
        )
    )
)

sample_event = _CANONICAL[
    "sample_event"
]

TRIGGER = "prv_actor_test_v1"


def h2_event():

    event = sample_event()

    payload = event["input"]

    payload[
        "contract_version"
    ] = H2_CONTRACT_VERSION

    payload[
        "trigger_token"
    ] = TRIGGER

    for sample in (
        payload[
            "dataset"
        ][
            "samples"
        ]
    ):
        sample[
            "prompt"
        ] = (
            f"{TRIGGER}, "
            "adult person, "
            "neutral identity reference"
        )

    training = payload[
        "training"
    ]

    training[
        "profile"
    ] = H2_TRAINING_PROFILE

    training[
        "width"
    ] = H2_WIDTH

    training[
        "height"
    ] = H2_HEIGHT

    return event


def test_h2_contract_accepts_recipe():

    request = parse_training_request(
        h2_event()
    )

    assert (
        request.payload[
            "contract_version"
        ] ==
        H2_CONTRACT_VERSION
    )

    assert (
        request.payload[
            "training"
        ][
            "profile"
        ] ==
        H2_TRAINING_PROFILE
    )

    assert (
        request.payload[
            "training"
        ][
            "width"
        ],
        request.payload[
            "training"
        ][
            "height"
        ],
    ) == (
        480,
        832,
    )


def test_h2_requires_trigger_in_every_prompt():

    event = h2_event()

    event[
        "input"
    ][
        "dataset"
    ][
        "samples"
    ][0][
        "prompt"
    ] = (
        "adult person without trigger"
    )

    with pytest.raises(
        WorkerError
    ) as error:

        parse_training_request(
            event
        )

    assert (
        error.value.code ==
        "H2_PROMPT_TRIGGER_MISSING"
    )


def test_h2_rejects_h1_geometry():

    event = h2_event()

    event[
        "input"
    ][
        "training"
    ][
        "width"
    ] = 832

    event[
        "input"
    ][
        "training"
    ][
        "height"
    ] = 480

    with pytest.raises(
        WorkerError
    ) as error:

        parse_training_request(
            event
        )

    assert (
        error.value.code ==
        "INVALID_DIT_TRAINING_PROFILE"
    )


def test_h1_contract_is_preserved():

    assert (
        CONTRACT_VERSION ==
        "privacy-identity-lora-training-v2"
    )

    assert (
        DIT_TRAINING_PROFILE ==
        "wan_dit_identity_video_v1"
    )
