import json
import sys

import pytest

from redisvl import __version__
from redisvl.cli.version import Version


def test_version_json_prints_package_version(monkeypatch, capsys):
    """``rvl version --json`` prints only a JSON object with the package version.

    Expected: stdout is a single JSON object with top-level key ``version`` and a
    string value equal to ``redisvl.__version__``; no other keys.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "version", "--json"])
    Version()
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert set(data.keys()) == {"version"}
    assert data["version"] == __version__


def test_version_short_prints_plain_version(monkeypatch, capsys):
    """``rvl version --short`` prints the bare version string (no JSON, no log prefix on stdout).

    Expected: stdout strip equals ``__version__`` only.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "version", "--short"])
    Version()
    assert capsys.readouterr().out.strip() == __version__


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(["--json", "--short"], id="json-then-short"),
        pytest.param(["--short", "--json"], id="short-then-json"),
    ],
)
def test_version_json_overrides_short(monkeypatch, capsys, extra):
    """If both ``--json`` and ``--short`` are set, JSON output wins regardless of order.

    Expected: same as ``--json`` alone: one JSON object ``{"version": <package>}``.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "version", *extra])
    Version()
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data == {"version": __version__}


def test_version_default_does_not_raise(monkeypatch):
    """``rvl version`` with no extra flags runs without raising (default human/log path).

    Expected: ``Version()`` completes; stdout may be empty while the version is logged.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "version"])
    Version()
