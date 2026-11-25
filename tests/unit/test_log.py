"""Logging reaches stderr when the application has not configured logging.

pytest's ``caplog`` fixture installs its own handler, so a test that uses it
passes even when ``get_logger`` swallows the record. These tests run in a
fresh interpreter and read stderr instead.
"""

import subprocess
import sys
import textwrap


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_warning_reaches_stderr_when_logging_is_unconfigured():
    result = _run("""
        import logging
        from redisvl.utils.log import get_logger

        get_logger("redisvl.demo").info("redisvl info hidden")
        get_logger("redisvl.demo").warning("from redisvl logger")
        logging.getLogger("plain.demo").warning("from plain logger")
        """)
    assert result.returncode == 0, result.stderr
    assert "redisvl info hidden" not in result.stderr
    assert "from redisvl logger" in result.stderr
    assert "from plain logger" in result.stderr


def test_configured_logging_receives_warning_exactly_once():
    result = _run("""
        import logging
        import sys

        logging.basicConfig(
            stream=sys.stdout,
            format="APP %(levelname)s %(name)s %(message)s",
            level=logging.WARNING,
        )
        from redisvl.utils.log import get_logger

        get_logger("redisvl.demo").warning("configured redisvl warning")
        """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "APP WARNING redisvl.demo configured redisvl warning"
    ]
    assert "configured redisvl warning" not in result.stderr
