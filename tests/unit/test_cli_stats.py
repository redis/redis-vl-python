import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from redisvl.cli.stats import STATS_KEYS, Stats, _stats_rows
from redisvl.exceptions import RedisSearchError
from redisvl.index import SearchIndex


@pytest.mark.parametrize("argv", [["rvl", "stats", "--help"], ["rvl", "stats", "-h"]])
def test_rvl_stats_help(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl stats --help`` and ``-h`` are discoverable.

    Expected behavior: ``SystemExit`` code is 0, stdout is non-empty, and
    stderr is empty.
    """
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Stats()
    captured = capsys.readouterr()

    # Help requests terminate successfully.
    assert exc_info.value.code == 0
    # Help is rendered to stdout, not stderr - critical for shell redirection.
    assert captured.out != ""
    # Successful help output does not leak to stderr.
    assert captured.err == ""


def test_rvl_stats_subprocess_help():
    """Tests that ``rvl stats --help`` works via the runner module.

    Expected behavior: the subprocess exits 0, stdout is non-empty, and
    stderr is empty.
    """
    result = subprocess.run(
        [sys.executable, "-m", "redisvl.cli.runner", "stats", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    # Process exited cleanly.
    assert result.returncode == 0
    # Help is rendered to stdout, not stderr.
    assert result.stdout != ""
    # Help is not emitted on stderr.
    assert result.stderr == ""


@pytest.fixture
def cli_stats_index(redis_url, redis_test_name):
    """Create a real temporary Redis index for stats CLI output tests."""
    index = SearchIndex.from_dict(
        {
            "index": {
                "name": redis_test_name("cli_stats_index"),
                "prefix": redis_test_name("cli_stats_doc"),
                "storage_type": "hash",
            },
            "fields": [{"name": "body", "type": "text"}],
        },
        redis_url=redis_url,
    )
    index.create(overwrite=True)
    try:
        yield index
    finally:
        index.delete(drop=True)


def _patch_search_index_for_stats(
    monkeypatch,
    *,
    info_error: BaseException | None = None,
    from_yaml_error: BaseException | None = None,
    index_name: str = "test-idx",
) -> None:
    """Patch ``SearchIndex`` to exercise ``rvl stats`` failure paths."""
    fake_index = SimpleNamespace(
        schema=SimpleNamespace(index=SimpleNamespace(name=index_name)),
        info=MagicMock(side_effect=info_error),
    )
    fake_search_index = MagicMock(return_value=fake_index)
    fake_search_index.from_yaml = MagicMock(
        side_effect=from_yaml_error, return_value=fake_index
    )
    monkeypatch.setattr("redisvl.cli.stats.SearchIndex", fake_search_index)


def test_stats_json(monkeypatch, capsys, redis_url, cli_stats_index):
    """Tests that ``rvl stats --json`` prints the documented JSON contract.

    Expected behavior: no ``SystemExit`` is raised, stdout is one parseable
    JSON line whose keys are exactly ``STATS_KEYS`` in order, native value
    types from Redis are preserved, and stderr is empty.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rvl",
            "stats",
            "-i",
            cli_stats_index.schema.index.name,
            "--json",
            "--url",
            redis_url,
        ],
    )

    Stats()
    captured = capsys.readouterr()
    out = captured.out.strip()

    # Single machine-readable line, nothing else on stdout.
    assert out.count("\n") == 0
    payload = json.loads(out)
    # Stable top-level JSON contract: every STATS_KEY in order, no extras.
    assert list(payload) == list(STATS_KEYS)
    # Native Redis stat types are preserved for machine consumers.
    assert isinstance(payload["num_docs"], int)
    # JSON success path is silent on stderr.
    assert captured.err == ""


def test_stats_table(monkeypatch, capsys, redis_url, cli_stats_index):
    """Tests that default ``rvl stats`` runs the human-readable path to completion.

    Expected behavior: no ``SystemExit`` is raised, stdout is non-empty
    (the table renderer ran), and stderr is empty.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rvl",
            "stats",
            "-i",
            cli_stats_index.schema.index.name,
            "--url",
            redis_url,
        ],
    )
    Stats()
    captured = capsys.readouterr()
    assert captured.out != ""
    assert captured.err == ""


def test_stats_rows_shape():
    """Tests that ``_stats_rows`` produces the documented row contract.

    Expected behavior: for an empty ``index_info`` dict, the helper returns
    ordered ``(key, None)`` pairs whose keys are exactly ``STATS_KEYS`` in
    order - the contract the JSON output relies on.
    """
    rows = _stats_rows({})
    data = dict(rows)

    # Every STATS_KEY appears, in declared order.
    assert list(data.keys()) == list(STATS_KEYS)
    # Missing input keys become None at this layer (serialized as JSON null).
    assert all(data[k] is None for k in STATS_KEYS)


@pytest.mark.parametrize(
    ("argv", "error_path"),
    [
        # Missing target selection.
        pytest.param(["rvl", "stats"], None, id="missing-target"),
        pytest.param(["rvl", "stats", "--json"], None, id="missing-target-json"),
        # Schema input errors.
        pytest.param(
            ["rvl", "stats", "-s", "/does/not/exist.yaml"],
            "schema",
            id="schema-input-error",
        ),
    ],
)
def test_stats_usage_errors(
    monkeypatch, capsys, argv: list[str], error_path: str | None
):
    """Tests that stats usage/input errors are reported consistently.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is non-empty.
    """
    if error_path == "schema":
        _patch_search_index_for_stats(
            monkeypatch,
            from_yaml_error=FileNotFoundError(
                "schema file missing: /does/not/exist.yaml"
            ),
        )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Stats()
    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert captured.out == ""
    assert captured.err != ""


@pytest.mark.parametrize(
    ("argv", "info_error"),
    [
        # Redis-side failures.
        pytest.param(
            ["rvl", "stats", "-i", "test-idx"],
            RedisSearchError("Unknown index name"),
            id="redis-search-error",
        ),
        pytest.param(
            ["rvl", "stats", "-i", "test-idx", "--json"],
            RedisSearchError("Unknown index name"),
            id="redis-search-error-json",
        ),
        # Generic runtime failures.
        pytest.param(
            ["rvl", "stats", "-i", "test-idx"],
            RuntimeError("boom"),
            id="runtime-error",
        ),
        pytest.param(
            ["rvl", "stats", "-i", "test-idx", "--json"],
            RuntimeError("boom"),
            id="runtime-error-json",
        ),
    ],
)
def test_stats_runtime_errors(
    monkeypatch, capsys, argv: list[str], info_error: BaseException
):
    """Tests that stats runtime/Redis failures are reported consistently.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty, and stderr
    is non-empty.
    """
    _patch_search_index_for_stats(monkeypatch, info_error=info_error)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Stats()
    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err != ""
