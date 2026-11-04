"""Logging reaches stderr when the application has not configured logging.

pytest's ``caplog`` fixture installs its own handler, so a test that uses it
passes even when ``get_logger`` swallows the record. These tests run in a
fresh interpreter and read stderr instead.
"""

import subprocess
import sys
import textwrap

from redisvl.utils.log import _DefaultStderrHandler, get_logger


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def _line_with(lines: list[str], message: str) -> str:
    matched = [line for line in lines if message in line]
    assert len(matched) == 1, lines
    return matched[0]


def _same_handler_format(left: str, right: str) -> None:
    """Both lines come from the same handler when only name and message differ."""

    def normalize(line: str, name: str, message: str) -> str:
        return line.replace(name, "NAME", 1).replace(message, "MSG", 1)

    assert normalize(left, "redisvl.demo", "from redisvl logger") == normalize(
        right, "plain.demo", "from plain logger"
    )


def test_get_logger_attaches_default_stderr_handler():
    name = "redisvl.test.stderr_handler"
    logger = get_logger(name)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], _DefaultStderrHandler)
    assert logger.propagate is True


def test_warning_reaches_stderr_without_logging_configuration():
    result = _run("""
        import logging
        from redisvl.utils.log import get_logger

        logger = get_logger("redisvl.demo")
        logger.info("redisvl info hidden")
        logger.warning("from redisvl logger")
        logger.error("from redisvl error")
        logging.getLogger("plain.demo").warning("from plain logger")
        """)
    assert result.returncode == 0, result.stderr
    assert "redisvl info hidden" not in result.stderr
    stderr_lines = result.stderr.splitlines()
    redis_line = _line_with(stderr_lines, "from redisvl logger")
    _line_with(stderr_lines, "from redisvl error")
    plain_line = _line_with(stderr_lines, "from plain logger")
    _same_handler_format(redis_line, plain_line)


def test_error_with_exc_info_does_not_write_traceback_to_stderr():
    result = _run("""
        from redisvl.utils.log import get_logger

        logger = get_logger("redisvl.cli.test")
        try:
            raise RuntimeError("cli failure")
        except RuntimeError as exc:
            logger.error(exc, exc_info=True)
        """)
    assert result.returncode == 0, result.stderr
    assert "cli failure" in result.stderr
    assert "Traceback" not in result.stderr


def test_info_propagates_when_root_logging_is_configured_for_debug():
    result = _run("""
        import logging
        import sys

        logging.basicConfig(
            stream=sys.stdout,
            format="APP %(levelname)s %(name)s %(message)s",
            level=logging.DEBUG,
        )
        from redisvl.utils.log import get_logger

        get_logger("redisvl.demo").info("configured redisvl info")
        """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "APP INFO redisvl.demo configured redisvl info"
    ]
    assert "configured redisvl info" not in result.stderr


def test_parent_logger_handlers_receive_redisvl_records():
    result = _run("""
        import logging
        import sys

        parent = logging.getLogger("redisvl")
        parent.setLevel(logging.INFO)
        parent.addHandler(
            logging.StreamHandler(sys.stdout)
        )
        parent.handlers[0].setFormatter(
            logging.Formatter("PARENT %(levelname)s %(name)s %(message)s")
        )
        from redisvl.utils.log import get_logger

        get_logger("redisvl.demo").info("parent handler info")
        """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "PARENT INFO redisvl.demo parent handler info"
    ]
    assert "parent handler info" not in result.stderr


def test_configured_logging_still_receives_redisvl_warnings():
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


def test_parent_and_child_default_handlers_do_not_duplicate_stderr():
    result = _run("""
        from redisvl.utils.log import get_logger

        get_logger("redisvl")
        get_logger("redisvl.demo").warning("single parent child warning")
        """)
    assert result.returncode == 0, result.stderr
    _line_with(result.stderr.splitlines(), "single parent child warning")


def test_warning_emit_succeeds_when_last_resort_is_none():
    result = _run("""
        import logging
        from redisvl.utils.log import get_logger

        logging.lastResort = None
        get_logger("redisvl.demo").warning("suppressed by none last resort")
        """)
    assert result.returncode == 0, result.stderr
    assert "suppressed by none last resort" not in result.stderr
    assert "Traceback" not in result.stderr
