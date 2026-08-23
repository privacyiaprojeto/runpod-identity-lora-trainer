from pathlib import Path
import re


TRAINER = Path("identity_worker/trainer.py")


def test_pod7d_accelerate_launch_is_explicit_single_process():
    text = TRAINER.read_text(encoding="utf-8")

    assert re.search(
        r"""["']accelerate["']\s*,\s*
            ["']launch["']\s*,\s*
            ["']--num_processes["']\s*,\s*
            ["']1["']""",
        text,
        re.VERBOSE | re.DOTALL,
    )

    assert text.count("--num_processes") == 1

    assert not re.search(
        r"""["']--multi_gpu["']""",
        text,
    )

    assert not re.search(
        r"""["']--num_processes["']\s*,\s*["'](?:[2-9]\d*)["']""",
        text,
    )

    assert not re.search(
        r"""["']--num_processes=(?:[2-9]\d*)["']""",
        text,
    )

    assert not re.search(
        r"""["']--num_machines["']\s*,\s*["'](?:[2-9]\d*)["']""",
        text,
    )

    assert not re.search(
        r"""["']--num_machines=(?:[2-9]\d*)["']""",
        text,
    )
