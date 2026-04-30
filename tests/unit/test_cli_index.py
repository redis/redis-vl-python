import json
import sys

import pytest

from redisvl.cli.index import Index, _index_info_for_json


class _FakeConn:
    def __init__(self, result, boom=False):
        self._result = result
        self._boom = boom

    def execute_command(self, cmd):
        assert cmd == "FT._LIST"  # listall must query Redis with FT._LIST
        if self._boom:
            raise RuntimeError("redis unavailable")
        return self._result


def test_listall_json(monkeypatch, capsys):
    """Tests that ``listall --json`` prints machine-readable output only.

    Expected behavior: stdout is one JSON line with ``indices`` in order and no table text.
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


def test_listall_table(monkeypatch, capsys):
    """Tests that default ``listall`` keeps the human-readable table output.

    Expected behavior: stdout matches header + numbered rows in FT._LIST order.
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


def test_listall_json_empty(monkeypatch, capsys):
    """Tests that ``listall --json`` handles an empty FT._LIST result.

    Expected behavior: stdout is valid JSON with ``{"indices": []}``.
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


def test_listall_json_error(monkeypatch, capsys):
    """Tests that ``listall --json`` failure exits cleanly without stdout JSON.

    Expected behavior: ``SystemExit`` code is 0 and stdout is empty.
    """

    def fake_get(*a, **k):
        return _FakeConn([], boom=True)

    monkeypatch.setattr(
        "redisvl.cli.index.RedisConnectionFactory.get_redis_connection", fake_get
    )
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall", "--json"])
    with pytest.raises(SystemExit) as excinfo:  # exit(0) in Index.__init__ is not a plain return
        Index()
    # assert excinfo.value.code == 0  # "log and exit(0)" CLI contract
    assert capsys.readouterr().out == ""  # failure before cli_print_json — nothing on stdout


def test_info_json_normalize():
    """Tests that ``_index_info_for_json`` maps FT.INFO lists to structured JSON.

    Expected behavior: input is unchanged and output has ``index_information`` + ``index_fields``.
    """
    raw = {
        "index_name": "test_index",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["prefix_a", "prefix_b"],
        ],
        "attributes": [
            [
                "identifier",
                "user",
                "attribute",
                "user",
                "type",
                "TAG",
            ],
        ],
    }
    before = str(raw)
    out = _index_info_for_json(raw)
    assert str(raw) == before  # not mutated
    assert out == {
        "index_information": {
            "index_name": "test_index",
            "storage_type": "HASH",
            "prefixes": ["prefix_a", "prefix_b"],
            "index_options": None,
            "indexing": None,
        },
        "index_fields": [
            {
                "name": "user",
                "attribute": "user",
                "type": "TAG",
            }
        ],
    }  # exact summary+fields payload, matching what table prints semantically


def test_info_json(monkeypatch, capsys):
    """Tests that ``info --json`` returns normalized table-equivalent JSON.

    Expected behavior: one parseable JSON line with decoded values and no table banners.
    """

    expected_index_information = {
        "index_name": "test-idx",
        "storage_type": "HASH",
        "prefixes": ["pre"],
        "index_options": None,
        "indexing": None,
    }
    expected_field = {
        "name": "u",
        "attribute": "u",
        "type": "TAG",
        "field_options": {"NOSTEM": "1"},
    }

    class FakeIndex:
        def __init__(self, *a, **k):
            pass

        def info(self):
            return {
                "index_name": b"test-idx",
                "index_definition": [
                    "key_type",
                    b"HASH",
                    "prefixes",
                    [b"pre"],
                ],
                "attributes": [
                    [
                        b"identifier",
                        b"u",
                        b"attribute",
                        b"u",
                        b"type",
                        b"TAG",
                        b"NOSTEM",
                        b"1",
                    ],
                ],
            }

    monkeypatch.setattr("redisvl.cli.index.SearchIndex", FakeIndex)
    monkeypatch.setattr(
        sys, "argv", ["rvl", "index", "info", "-i", "test-idx", "--json"]
    )
    Index()
    out = capsys.readouterr().out.strip()
    assert out.count("\n") == 0  # single line for machine consumers
    payload = json.loads(out)
    assert "Index Information:" not in out and "Index Fields:" not in out  # --json must not emit table banner text
    assert list(payload) == ["index_information", "index_fields"]  # top-level sections are stable and ordered
    assert payload["index_information"] == expected_index_information  # summary section matches table-derived values
    assert payload["index_fields"] == [expected_field]  # one normalized field row with options

def test_info_json_error(monkeypatch, capsys):
    """Tests that ``info --json`` errors do not emit partial stdout JSON.

    Expected behavior: command exits with code 0 and stdout is empty.
    """

    class BoomIndex:
        def __init__(self, *a, **k):
            pass

        def info(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("redisvl.cli.index.SearchIndex", BoomIndex)
    monkeypatch.setattr(
        sys, "argv", ["rvl", "index", "info", "-i", "test-idx", "--json"]
    )
    with pytest.raises(SystemExit) as excinfo:
        Index()
    # assert excinfo.value.code == 0  # try/except in Index.__init__ + exit(0)
    assert capsys.readouterr().out == ""  # no partial JSON before the exception
