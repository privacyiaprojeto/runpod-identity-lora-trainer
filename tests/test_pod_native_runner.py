from __future__ import annotations

import ast
import importlib.util
import json

from pathlib import Path

import pytest


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

HANDLER = ROOT / "handler.py"
RUNNER = ROOT / "pod_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "privacy_pod_runner_test_module",
        RUNNER,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None

    spec.loader.exec_module(
        module
    )

    return module


def _valid_contract(
    actor: str,
    run: str,
) -> dict:

    return {
        "contract_version":
            "privacy-identity-lora-training-v3",

        "actor_profile_id":
            actor,

        "training_run_id":
            run,

        "training": {
            "profile":
                "wan_dit_identity_video_h2_v1",

            "automatic_retry":
                False,
        },

        "smoke": {
            "one_shot":
                True,

            "max_jobs":
                1,
        },

        "safety": {
            "automatic_retry_allowed":
                False,

            "inference_injection_allowed":
                False,
        },

        "output": {
            "public":
                False,
        },
    }


def test_serverless_start_is_main_guarded():
    source = HANDLER.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(
                node,
                ast.Call,
            )
            and
            ast.unparse(
                node.func
            )
            ==
            "runpod.serverless.start"
        )
    ]

    assert len(calls) == 1

    assert (
        "def _start_serverless_transport():"
        in source
    )

    assert (
        "if __name__ == '__main__':"
        in source
    )

    assert (
        "runpod.serverless.start({'handler': handler})"
        in source
    )


def test_native_runner_has_no_serverless_start():
    source = RUNNER.read_text(
        encoding="utf-8"
    )

    assert (
        "runpod.serverless.start"
        not in source
    )

    assert (
        "_handle_training"
        in source
    )

    assert (
        "POD_NATIVE_AUTOMATIC_RETRY_FORBIDDEN"
        in source
    )


def test_native_runner_closed_by_default(
    tmp_path,
    monkeypatch,
):
    runner = _load_runner()

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_RUNTIME_ROOT",
        str(
            tmp_path
        ),
    )

    monkeypatch.delenv(
        "PRIVACY_IDENTITY_POD_NATIVE_ENABLED",
        raising=False,
    )

    request = (
        tmp_path /
        "request.json"
    )

    request.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="POD_NATIVE_DISABLED",
    ):
        runner._load_and_validate_event(
            request
        )


def test_native_runner_accepts_exact_h2_scope(
    tmp_path,
    monkeypatch,
):
    runner = _load_runner()

    actor = (
        "767f0277-6b78-4afb-8b5e-88b302033705"
    )

    run = (
        "bfdb9e6c-051b-4300-a306-6e0f38bdbc94"
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_RUNTIME_ROOT",
        str(
            tmp_path
        ),
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_NATIVE_ENABLED",
        "true",
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_MAX_JOBS",
        "1",
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_AUTOMATIC_RETRY",
        "false",
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_ACTOR_PROFILE_ID",
        actor,
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_TRAINING_RUN_ID",
        run,
    )

    request = (
        tmp_path /
        "request.json"
    )

    request.write_text(
        json.dumps(
            _valid_contract(
                actor,
                run,
            )
        ),

        encoding="utf-8",
    )

    event = runner._load_and_validate_event(
        request
    )

    assert (
        event["input"]["actor_profile_id"]
        ==
        actor
    )

    assert (
        event["input"]["training_run_id"]
        ==
        run
    )


def test_native_runner_rejects_wrong_actor(
    tmp_path,
    monkeypatch,
):
    runner = _load_runner()

    actor = (
        "767f0277-6b78-4afb-8b5e-88b302033705"
    )

    run = (
        "bfdb9e6c-051b-4300-a306-6e0f38bdbc94"
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_RUNTIME_ROOT",
        str(
            tmp_path
        ),
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_NATIVE_ENABLED",
        "true",
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_MAX_JOBS",
        "1",
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_AUTOMATIC_RETRY",
        "false",
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_ACTOR_PROFILE_ID",
        actor,
    )

    monkeypatch.setenv(
        "PRIVACY_IDENTITY_POD_TRAINING_RUN_ID",
        run,
    )

    request = (
        tmp_path /
        "request.json"
    )

    request.write_text(
        json.dumps(
            _valid_contract(
                "00000000-0000-0000-0000-000000000000",
                run,
            )
        ),

        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="POD_NATIVE_ACTOR_SCOPE_MISMATCH",
    ):
        runner._load_and_validate_event(
            request
        )