import json
import sys

import pytest

from redisvl.cli.stats import STATS_KEYS, Stats, _stats_rows


def test_stats_rows_includes_all_stable_top_level_keys_in_order():
    """``_stats_rows({})`` returns the full ordered row list for an empty index info.

    Expected behavior: produces a complete, ``STATS_KEYS``-ordered set of rows
    regardless of input; preserves value types at this layer; and represents
    missing keys as ``None`` so JSON output remains machine-readable.
    """
    data = dict(_stats_rows({}))
    assert list(data.keys()) == list(STATS_KEYS)  # column order matches STATS_KEYS
    assert all(data[k] is None for k in STATS_KEYS)  # missing index_info keys -> None


def test_stats_json_prints_only_json_to_stdout(monkeypatch, capsys):
    """``rvl stats -i <name> --json`` writes only a JSON object to stdout.

    Uses a fake ``SearchIndex`` so no Redis is required.

    Expected behavior: ``--json`` skips ``_display_stats`` and emits one
    single-line JSON document with the full ``STATS_KEYS`` schema and
    native values (e.g. ``num_docs=7`` -> ``7``).

    Row order is covered by ``test_stats_rows_*``; JSON key order is covered
    by ``test_cli_print_json_preserves_key_order``.
    """

    class FakeIndex:
        def __init__(self, *a, **k):
            pass

        def info(self):
            return {"num_docs": 7}

    monkeypatch.setattr("redisvl.cli.stats.SearchIndex", FakeIndex)
    monkeypatch.setattr(sys, "argv", ["rvl", "stats", "-i", "test-idx", "--json"])
    Stats()
    out = capsys.readouterr().out.strip()
    assert "Statistics" not in out  # --json must not emit the table UI text
    assert out.count("\n") == 0  # exactly one JSON object on stdout, no extra lines
    payload = json.loads(out)
    assert set(payload) == set(STATS_KEYS)  # same stat keys as the shared schema list
    assert payload["num_docs"] == 7  # numbers remain numbers for machine consumers
    assert payload["num_terms"] is None  # missing values become JSON null, not "None"


def test_stats_default_prints_table(monkeypatch, capsys):
    """``rvl stats -i <name>`` without ``--json`` still renders the ASCII table.

    Expected behavior: ``Stats.stats`` selects the human-readable branch and
    delegates to ``_display_stats``; the ``Statistics:`` banner is the signal
    that the table path ran. Guards against the ``--json`` plumbing regressing
    the default mode.
    """

    class FakeIndex:
        def __init__(self, *a, **k):
            pass

        def info(self):
            return {"num_docs": 1}

    monkeypatch.setattr("redisvl.cli.stats.SearchIndex", FakeIndex)
    monkeypatch.setattr(sys, "argv", ["rvl", "stats", "-i", "test-idx"])
    Stats()
    out = capsys.readouterr().out
    assert "Statistics:" in out  # non-JSON path prints the table header line


def test_stats_missing_index_and_schema_exits_zero_without_json(monkeypatch, capsys):
    """Without -i/-s, ``_connect_to_index`` logs and ``exit(0)`` s; no JSON leaks.

    Expected behavior: invalid input follows the standard ``rvl`` "log + exit 0"
    pattern. ``--json`` does not relax that contract — stdout stays empty so
    machine consumers never see a half-formed JSON object.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "stats", "--json"])
    with pytest.raises(SystemExit) as excinfo:
        Stats()
    assert excinfo.value.code == 2
    assert capsys.readouterr().out == ""  # no JSON object emitted on error


def test_stats_info_failure_exits_zero_without_json(monkeypatch, capsys):
    """If ``index.info()`` raises, ``Stats.__init__`` logs and ``exit(0)`` s; no JSON leaks.

    Expected behavior: ``try/except Exception`` converts backend failures into
    ``exit(0)`` (no traceback). With ``--json``, stdout stays empty so "exit 0
    + empty stdout" means "no result", never "malformed result".
    """

    class BoomIndex:
        def __init__(self, *a, **k):
            pass

        def info(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("redisvl.cli.stats.SearchIndex", BoomIndex)
    monkeypatch.setattr(sys, "argv", ["rvl", "stats", "-i", "test-idx", "--json"])
    with pytest.raises(SystemExit) as excinfo:
        Stats()
    assert excinfo.value.code == 1
    assert capsys.readouterr().out == ""
