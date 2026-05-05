import re
import subprocess
import sys

import pytest

from redisvl.cli.main import RedisVlCLI

_COMMANDS = ("index", "mcp", "version", "stats")


def _assert_help_contract(help_text: str) -> None:
    """Assert that ``rvl`` help lists every supported top-level command."""
    for name in _COMMANDS:
        # Each supported top-level command appears on its own help line.
        assert re.search(rf"^\s*{re.escape(name)}\s+", help_text, re.MULTILINE)


@pytest.mark.parametrize("argv", [["rvl"], ["rvl", "--help"], ["rvl", "-h"]])
def test_rvl_help(monkeypatch, capsys, argv: list[str]):
    """Help paths (`rvl`, `--help`, `-h`) exit 0 and print to stdout."""
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        RedisVlCLI()
    out = capsys.readouterr()

    # Help requests terminate successfully.
    assert exc_info.value.code == 0

    # Successful help output does not leak to stderr.
    assert out.err == ""

    # stdout contains the expected top-level help contract.
    _assert_help_contract(out.out)


def test_unknown_command(monkeypatch, capsys):
    """Unknown commands exit 2, write error/help to stderr, and keep stdout empty."""
    monkeypatch.setattr(sys, "argv", ["rvl", "notacommand"])

    with pytest.raises(SystemExit) as exc_info:
        RedisVlCLI()
    out = capsys.readouterr()

    # Unknown commands use the CLI usage-error exit code.
    assert exc_info.value.code == 2

    # stdout stays empty on this error path.
    assert out.out == ""

    # stderr identifies the rejected command token.
    assert "Unknown command: notacommand" in out.err
    for name in _COMMANDS:
        # stderr help still lists every valid top-level command.
        assert name in out.err


def test_subprocess_module_help():
    """Run ``python -m redisvl.cli.runner --help`` and verify it exits 0 with help on stdout.

    Acts as an end-to-end check that the installed CLI entrypoint actually works.
    """
    result = subprocess.run(
        [sys.executable, "-m", "redisvl.cli.runner", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    # Help subprocess exits successfully.
    assert result.returncode == 0
    # No stderr output for help.
    assert result.stderr == ""
    # Stdout includes the same help contract as in-process tests.
    _assert_help_contract(result.stdout)
