import json
import sys

import pytest

from redisvl.cli.index import Index


class _FakeConn:
    def __init__(self, result, boom=False):
        self._result = result
        self._boom = boom

    def execute_command(self, cmd):
        assert cmd == "FT._LIST"  # listall must query Redis with FT._LIST
        if self._boom:
            raise RuntimeError("redis unavailable")
        return self._result


def test_index_listall_json_prints_single_object(monkeypatch, capsys):
    """Mocked successful ``FT._LIST`` in ``--json`` mode.

    What: ``listall`` uses ``cli_print_json`` and skips the text banner / loop.

    Expected behavior: exactly one line of parseable JSON on stdout with key
    ``indices``; values are the ``convert_bytes`` result of the mock list, in
    order; no ``Indices:`` or partial human output; no extra newlines.
    """

    def fake_get(*a, **k):
        return _FakeConn([b"idx_a", b"idx_b"])

    monkeypatch.setattr(
        "redisvl.cli.index.RedisConnectionFactory.get_redis_connection", fake_get
    )
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall", "--json"])
    Index()
    out = capsys.readouterr().out.strip()
    assert "Indices:" not in out  # --json must not print the human banner
    assert out.count("\n") == 0  # single machine-readable line, nothing else on stdout
    payload = json.loads(out)
    assert payload == {"indices": ["idx_a", "idx_b"]}  # same order/encoding as table path would show


def test_index_listall_table_prints_banner(monkeypatch, capsys):
    """Default ``listall`` (no ``--json``) uses the text formatter.

    What: non-``--json`` still prints ``Indices:`` and a numbered list from the
    same mocked ``FT._LIST`` return.

    Expected behavior: stdout splits into a header line plus one ``N. name``
    line per index, in ``FT._LIST`` order, with no extra lines.
    """

    def fake_get(*a, **k):
        return _FakeConn([b"one", b"two"])

    monkeypatch.setattr(
        "redisvl.cli.index.RedisConnectionFactory.get_redis_connection", fake_get
    )
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall"])
    Index()
    out = capsys.readouterr().out
    lines = [ln.strip() for ln in out.strip().splitlines()]
    assert lines == [
        "Indices:",
        "1. one",
        "2. two",
    ]  # exact table output: header then rows matching mock order and labels


def test_index_listall_json_empty_indices(monkeypatch, capsys):
    """Empty result from ``FT._LIST`` in ``--json`` mode.

    What: empty list after ``convert_bytes`` still forms a valid JSON object.

    Expected behavior: one line ``{"indices": []}`` with no error.
    """

    def fake_get(*a, **k):
        return _FakeConn([])

    monkeypatch.setattr(
        "redisvl.cli.index.RedisConnectionFactory.get_redis_connection", fake_get
    )
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall", "--json"])
    Index()
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {"indices": []}  # empty array is a valid success payload


def test_index_listall_execute_error_exits_zero_without_json_stdout(monkeypatch, capsys):
    """``FT._LIST`` raises: ``Index`` catches, logs, ``exit(0)``; no JSON leak.

    What: backend failure is handled by the ``Index.__init__`` try/except like
    other rvl subcommands, even with ``--json`` requested.

    Expected behavior: process exits 0; stdout is empty (no half-written JSON
    from ``cli_print_json``).
    """

    def fake_get(*a, **k):
        return _FakeConn([], boom=True)

    monkeypatch.setattr(
        "redisvl.cli.index.RedisConnectionFactory.get_redis_connection", fake_get
    )
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall", "--json"])
    with pytest.raises(SystemExit) as excinfo:  # exit(0) in Index.__init__ is not a plain return
        Index()
    assert excinfo.value.code == 0  # "log and exit(0)" CLI contract
    assert capsys.readouterr().out == ""  # failure before cli_print_json — nothing on stdout
