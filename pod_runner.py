#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os

from pathlib import Path
from typing import Any


EXPECTED_CONTRACT = "privacy-identity-lora-training-v3"
EXPECTED_PROFILE = "wan_dit_identity_video_h2_v1"


def _truthy(name: str) -> bool:
    return (
        os.getenv(name, "")
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )


def _required_env(name: str) -> str:
    value = os.getenv(
        name,
        "",
    ).strip()

    if not value:
        raise RuntimeError(
            f"POD_NATIVE_REQUIRED_ENV_MISSING:{name}"
        )

    return value


def _runtime_root() -> Path:
    return Path(
        os.getenv(
            "PRIVACY_IDENTITY_POD_RUNTIME_ROOT",
            "/workspace",
        )
    ).expanduser().resolve()


def _require_inside_root(
    path: Path,
    root: Path,
    code: str,
) -> Path:

    resolved = (
        path
        .expanduser()
        .resolve()
    )

    try:
        resolved.relative_to(
            root
        )

    except ValueError as exc:
        raise RuntimeError(
            f"{code}:{resolved}"
        ) from exc

    return resolved


def _assert_native_gate() -> None:
    if not _truthy(
        "PRIVACY_IDENTITY_POD_NATIVE_ENABLED"
    ):
        raise RuntimeError(
            "POD_NATIVE_DISABLED"
        )

    if (
        os.getenv(
            "PRIVACY_IDENTITY_POD_MAX_JOBS",
            "",
        ).strip()
        !=
        "1"
    ):
        raise RuntimeError(
            "POD_NATIVE_MAX_JOBS_MUST_BE_ONE"
        )

    if _truthy(
        "PRIVACY_IDENTITY_POD_AUTOMATIC_RETRY"
    ):
        raise RuntimeError(
            "POD_NATIVE_AUTOMATIC_RETRY_FORBIDDEN"
        )


def _load_and_validate_event(
    request_file: Path,
) -> dict[str, Any]:

    _assert_native_gate()

    root = _runtime_root()

    path = _require_inside_root(
        request_file,
        root,
        "POD_NATIVE_REQUEST_OUTSIDE_RUNTIME_ROOT",
    )

    if not path.is_file():
        raise RuntimeError(
            f"POD_NATIVE_REQUEST_NOT_FOUND:{path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "POD_NATIVE_REQUEST_JSON_INVALID"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "POD_NATIVE_REQUEST_MUST_BE_OBJECT"
        )

    event = (
        payload
        if "input" in payload
        else {
            "input": payload
        }
    )

    contract = event.get(
        "input"
    )

    if not isinstance(
        contract,
        dict,
    ):
        raise RuntimeError(
            "POD_NATIVE_INPUT_MUST_BE_OBJECT"
        )

    if (
        contract.get(
            "contract_version"
        )
        !=
        EXPECTED_CONTRACT
    ):
        raise RuntimeError(
            "POD_NATIVE_CONTRACT_V3_REQUIRED"
        )

    training = contract.get(
        "training"
    )

    if not isinstance(
        training,
        dict,
    ):
        raise RuntimeError(
            "POD_NATIVE_TRAINING_BLOCK_REQUIRED"
        )

    if (
        training.get(
            "profile"
        )
        !=
        EXPECTED_PROFILE
    ):
        raise RuntimeError(
            "POD_NATIVE_H2_PROFILE_REQUIRED"
        )

    if (
        training.get(
            "automatic_retry"
        )
        is not False
    ):
        raise RuntimeError(
            "POD_NATIVE_CONTRACT_AUTOMATIC_RETRY_FORBIDDEN"
        )

    expected_actor = _required_env(
        "PRIVACY_IDENTITY_POD_ACTOR_PROFILE_ID"
    )

    expected_run = _required_env(
        "PRIVACY_IDENTITY_POD_TRAINING_RUN_ID"
    )

    if (
        contract.get(
            "actor_profile_id"
        )
        !=
        expected_actor
    ):
        raise RuntimeError(
            "POD_NATIVE_ACTOR_SCOPE_MISMATCH"
        )

    if (
        contract.get(
            "training_run_id"
        )
        !=
        expected_run
    ):
        raise RuntimeError(
            "POD_NATIVE_TRAINING_RUN_SCOPE_MISMATCH"
        )

    smoke = contract.get(
        "smoke"
    )

    if not isinstance(
        smoke,
        dict,
    ):
        raise RuntimeError(
            "POD_NATIVE_SMOKE_BLOCK_REQUIRED"
        )

    if (
        smoke.get(
            "one_shot"
        )
        is not True
    ):
        raise RuntimeError(
            "POD_NATIVE_ONE_SHOT_REQUIRED"
        )

    if (
        int(
            smoke.get(
                "max_jobs",
                0,
            )
        )
        !=
        1
    ):
        raise RuntimeError(
            "POD_NATIVE_CONTRACT_MAX_JOBS_MUST_BE_ONE"
        )

    safety = contract.get(
        "safety"
    )

    if not isinstance(
        safety,
        dict,
    ):
        raise RuntimeError(
            "POD_NATIVE_SAFETY_BLOCK_REQUIRED"
        )

    if (
        safety.get(
            "automatic_retry_allowed"
        )
        is not False
    ):
        raise RuntimeError(
            "POD_NATIVE_SAFETY_RETRY_FORBIDDEN"
        )

    if (
        safety.get(
            "inference_injection_allowed"
        )
        is not False
    ):
        raise RuntimeError(
            "POD_NATIVE_INFERENCE_INJECTION_FORBIDDEN"
        )

    output = contract.get(
        "output"
    )

    if not isinstance(
        output,
        dict,
    ):
        raise RuntimeError(
            "POD_NATIVE_OUTPUT_BLOCK_REQUIRED"
        )

    if (
        output.get(
            "public"
        )
        is not False
    ):
        raise RuntimeError(
            "POD_NATIVE_PUBLIC_OUTPUT_FORBIDDEN"
        )

    return event


def _invoke_training(
    event: dict[str, Any],
) -> Any:

    handler_module = importlib.import_module(
        "handler"
    )

    handle_training = getattr(
        handler_module,
        "_handle_training",
        None,
    )

    if not callable(
        handle_training
    ):
        raise RuntimeError(
            "POD_NATIVE_HANDLE_TRAINING_NOT_AVAILABLE"
        )

    return handle_training(
        event
    )


def _write_result(
    result_file: Path,
    result: Any,
) -> None:

    root = _runtime_root()

    target = _require_inside_root(
        result_file,
        root,
        "POD_NATIVE_RESULT_OUTSIDE_RUNTIME_ROOT",
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = target.with_name(
        target.name +
        ".tmp"
    )

    temp.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",

        encoding="utf-8",
    )

    temp.replace(
        target
    )


def main(
    argv: list[str] | None = None,
) -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--request-file",
        required=True,
    )

    parser.add_argument(
        "--result-file",
        required=True,
    )

    args = parser.parse_args(
        argv
    )

    event = _load_and_validate_event(
        Path(
            args.request_file
        )
    )

    result = _invoke_training(
        event
    )

    _write_result(
        Path(
            args.result_file
        ),
        result,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )