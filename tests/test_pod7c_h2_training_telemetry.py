from __future__ import annotations

import importlib.util
import json
import runpy
from pathlib import Path

import pytest

from identity_worker.errors import WorkerError

from identity_worker.trainer import (
    TRAINING_TELEMETRY_SCHEMA,
    load_training_telemetry,
    training_telemetry_path,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def _write_valid_telemetry(
    root: Path,
) -> Path:

    path = training_telemetry_path(
        root
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for step in range(
        1,
        801,
    ):
        records.append(
            {
                "event":
                    "optimizer_step",

                "step":
                    step,

                "loss":
                    1.0 / (
                        step + 1
                    ),

                "learning_rate":
                    0.00005,

                "checkpoint":
                    step
                    in (
                        400,
                        600,
                        800,
                    ),
            }
        )

    records.append(
        {
            "event":
                "training_end",

            "step":
                800,

            "elapsed_seconds":
                123.456,

            "termination_reason":
                "optimizer_step_contract_reached",

            "checkpoint_steps":
                [
                    400,
                    600,
                    800,
                ],
        }
    )

    path.write_text(
        "\n".join(
            json.dumps(
                record,
                sort_keys=True,
            )
            for record
            in records
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def test_training_telemetry_receipt_is_exact():

    import tempfile

    with tempfile.TemporaryDirectory() as raw:

        output_dir = Path(
            raw
        )

        path = (
            _write_valid_telemetry(
                output_dir
            )
        )

        receipt = (
            load_training_telemetry(
                output_dir
            )
        )

        assert (
            receipt[
                "schema_version"
            ]
            ==
            TRAINING_TELEMETRY_SCHEMA
        )

        assert (
            receipt[
                "path"
            ]
            ==
            path
        )

        assert (
            receipt[
                "step_count"
            ]
            ==
            800
        )

        assert (
            receipt[
                "first_step"
            ],
            receipt[
                "last_step"
            ],
        ) == (
            1,
            800,
        )

        assert (
            receipt[
                "checkpoint_steps"
            ]
            ==
            [
                400,
                600,
                800,
            ]
        )

        assert (
            receipt[
                "termination_reason"
            ]
            ==
            "optimizer_step_contract_reached"
        )

        assert (
            len(
                receipt[
                    "sha256"
                ]
            )
            ==
            64
        )


def test_training_telemetry_blocks_step_gap():

    import tempfile

    with tempfile.TemporaryDirectory() as raw:

        output_dir = Path(
            raw
        )

        path = (
            _write_valid_telemetry(
                output_dir
            )
        )

        lines = (
            path.read_text(
                encoding="utf-8"
            )
            .splitlines()
        )

        broken = json.loads(
            lines[399]
        )

        broken[
            "step"
        ] = 401

        lines[399] = (
            json.dumps(
                broken,
                sort_keys=True,
            )
        )

        path.write_text(
            "\n".join(
                lines
            )
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(
            WorkerError
        ) as error:

            load_training_telemetry(
                output_dir
            )

        assert (
            error.value.code
            ==
            "TRAINING_TELEMETRY_STEP_INVALID"
        )


def test_runner_patch_emits_loss_lr_and_terminal_receipt():

    namespace = runpy.run_path(
        str(
            ROOT /
            "tests" /
            "test_diffsynth_runner_patcher.py"
        )
    )

    module = namespace[
        "MODULE"
    ]

    fixture = namespace[
        "OFFICIAL_TAIL_FIXTURE"
    ]

    patched, changed = (
        module.patch_runner_text(
            fixture
        )
    )

    assert changed is True

    assert (
        "PRIVACY_H2_TRAINING_TELEMETRY_V1"
        in patched
    )

    assert (
        "PRIVACY_TRAINING_TELEMETRY_PATH"
        in patched
    )

    assert (
        '"loss": privacy_loss_value'
        in patched
    )

    assert (
        '"learning_rate": privacy_learning_rate'
        in patched
    )

    assert (
        '"termination_reason":'
        in patched
    )

    assert (
        '"optimizer_step_contract_reached"'
        in patched
    )

    assert (
        'allow_nan=False'
        in patched
    )

    compile(
        patched,
        "runner.py",
        "exec",
    )


def test_trainer_exports_telemetry_path_to_runner():

    source = (
        ROOT /
        "identity_worker" /
        "trainer.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        '"PRIVACY_TRAINING_TELEMETRY_PATH"'
        in source
    )

    assert (
        "load_training_telemetry("
        in source
    )


def test_handler_preserves_h2_manifest_before_training():

    source = (
        ROOT /
        "handler.py"
    ).read_text(
        encoding="utf-8"
    )

    training = (
        source
        .split(
            "def _handle_training(",
            1,
        )[1]
        .split(
            "def _handle_preview(",
            1,
        )[0]
    )

    manifest_gate = (
        training.index(
            "H2_COMPILED_MANIFEST_MISSING"
        )
    )

    training_call = (
        training.index(
            "run_training("
        )
    )

    assert (
        manifest_gate <
        training_call
    )

    assert (
        "'compiled_training_manifest':"
        in training
    )

    assert (
        "'training_telemetry':"
        in training
    )

    assert (
        "'contract_version':\n"
        "                    contract_version"
        in training
    )

    assert (
        "compiled_manifest_sha256"
        in training
    )


def test_handler_no_longer_hardcodes_v2_training_response():

    source = (
        ROOT /
        "handler.py"
    ).read_text(
        encoding="utf-8"
    )

    training = (
        source
        .split(
            "def _handle_training(",
            1,
        )[1]
        .split(
            "def _handle_preview(",
            1,
        )[0]
    )

    assert (
        "'contract_version': CONTRACT_VERSION"
        not in training
    )
