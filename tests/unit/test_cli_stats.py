import json
import subprocess
import sys

import pytest

from redisvl.cli.stats import STATS_KEYS, Stats, _stats_rows


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
    """End-to-end smoke test of ``rvl stats --help`` via the runner module.

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


def _raise(exc: BaseException):
    """Return a zero-arg callable that raises ``exc``."""

    def _do():
        raise exc

    return _do


def _patch_search_index_for_stats(
    monkeypatch,
    *,
    info_behavior=None,
    index_name: str = "test-idx",
) -> None:
    """Patch ``redisvl.cli.stats.SearchIndex`` with a minimal fake for ``rvl stats`` tests.

    By default builds a happy-path FakeIndex whose ``info()`` returns
    ``{"num_docs": 7}`` - one populated stat key, the rest absent so the
    missing-key -> ``None`` path is exercised. Pass
    ``info_behavior=_raise(SomeError())`` to make ``info()`` raise instead.
    ``index_name`` is surfaced via ``index.schema.index.name`` so
    ``exit_redis_search_error`` can format its message verbatim.
    """
    if info_behavior is None:

        def info_behavior():
            return {"num_docs": 7}

    class _FakeIndexInfo:
        name = index_name

    class _FakeSchema:
        index = _FakeIndexInfo()

    class FakeIndex:
        schema = _FakeSchema()

        def __init__(self, *a, **k):
            # Absorb the ``schema=`` and ``redis_url=`` kwargs that
            # ``_connect_to_index`` passes to ``SearchIndex(...)`` directly.
            # Without this, Python's default ``__init__`` rejects them.
            pass

        @classmethod
        def from_yaml(cls, *_args, **_kwargs):
            return cls()

        def info(self):
            return info_behavior()

    monkeypatch.setattr("redisvl.cli.stats.SearchIndex", FakeIndex)


def _patch_search_index_from_yaml_raises(monkeypatch, exc: BaseException) -> None:
    """Patch ``redisvl.cli.stats.SearchIndex.from_yaml`` to raise ``exc``.

    Used by the schema-input-error path where ``-s <path>`` is provided but
    loading fails (e.g. file missing, malformed YAML).
    """

    class FakeSearchIndex:
        @classmethod
        def from_yaml(cls, *_args, **_kwargs):
            raise exc

    monkeypatch.setattr("redisvl.cli.stats.SearchIndex", FakeSearchIndex)


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "stats"],
        ["rvl", "stats", "--json"],
    ],
)
def test_stats_no_target(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl stats`` without ``-i`` or ``-s`` exits with a usage error.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is non-empty. ``--json`` is parametrized to confirm no JSON contract is
    invented for usage errors.
    """
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Stats()
    captured = capsys.readouterr()

    # _connect_to_index uses argparse-style usage-error exit code.
    assert exc_info.value.code == 2
    # Usage errors must not pollute stdout, even with --json.
    assert captured.out == ""
    # Some explanatory message reaches stderr; exact wording is not part of the contract.
    assert captured.err != ""


def test_stats_schema_input_error(monkeypatch, capsys):
    """Tests that ``rvl stats -s <bad>`` reports schema-load failures on stderr.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is non-empty.
    """
    _patch_search_index_from_yaml_raises(
        monkeypatch, FileNotFoundError("schema file missing: /does/not/exist.yaml")
    )
    monkeypatch.setattr(sys, "argv", ["rvl", "stats", "-s", "/does/not/exist.yaml"])

    with pytest.raises(SystemExit) as exc_info:
        Stats()
    captured = capsys.readouterr()

    # exit_schema_input_error uses exit code 2 when -s was provided.
    assert exc_info.value.code == 2
    # Nothing on stdout when schema input fails.
    assert captured.out == ""
    # The failure is surfaced on stderr.
    assert captured.err != ""


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "stats", "-i", "test-idx"],
        ["rvl", "stats", "-i", "test-idx", "--json"],
    ],
)
def test_stats_redis_search_error(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl stats`` reports Redis-side failures on stderr.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty, and stderr
    is non-empty. ``--json`` is parametrized to confirm the contract is
    uniform.
    """
    from redisvl.exceptions import RedisSearchError

    _patch_search_index_for_stats(
        monkeypatch, info_behavior=_raise(RedisSearchError("Unknown index name"))
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Stats()
    captured = capsys.readouterr()

    # Documented Redis-failure exit code from exit_redis_search_error.
    assert exc_info.value.code == 1
    # No partial output before the failure.
    assert captured.out == ""
    # The failure is surfaced on stderr.
    assert captured.err != ""


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "stats", "-i", "test-idx"],
        ["rvl", "stats", "-i", "test-idx", "--json"],
    ],
)
def test_stats_runtime_error(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl stats`` reports runtime failures on stderr.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty (no
    half-formed output), and stderr is non-empty. ``--json`` is parametrized
    to confirm the contract is uniform.
    """
    _patch_search_index_for_stats(
        monkeypatch, info_behavior=_raise(RuntimeError("boom"))
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Stats()
    captured = capsys.readouterr()

    # Generic runtime failure exits 1 from __init__'s catch-all.
    assert exc_info.value.code == 1
    # Failure happens before any rendering - nothing on stdout.
    assert captured.out == ""
    # The failure is surfaced on stderr; exact wording is not part of the contract.
    assert captured.err != ""


def test_stats_json(monkeypatch, capsys):
    """Tests that ``rvl stats --json`` prints the documented JSON contract.

    Expected behavior: no ``SystemExit`` is raised, stdout is one parseable
    JSON line whose keys are exactly ``STATS_KEYS`` in order, native value
    types are preserved, and missing input keys become JSON ``null``.
    Stderr is empty.
    """
    _patch_search_index_for_stats(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rvl", "stats", "-i", "test-idx", "--json"])

    try:
        Stats()
    except SystemExit as exc:
        # Success path must return cleanly, not call sys.exit.
        pytest.fail(f"stats --json raised SystemExit({exc.code}) on the success path")
    captured = capsys.readouterr()
    out = captured.out.strip()

    # Single machine-readable line, nothing else on stdout.
    assert out.count("\n") == 0
    payload = json.loads(out)
    # Stable top-level JSON contract: every STATS_KEY in order, no extras.
    assert list(payload) == list(STATS_KEYS)
    # Native types are preserved for machine consumers.
    assert payload["num_docs"] == 7
    # Missing input keys deserialize to None and is not mistyped.
    assert payload["num_terms"] is None
    # JSON success path is silent on stderr.
    assert captured.err == ""


def test_stats_table(monkeypatch, capsys):
    """Tests that default ``rvl stats`` runs the human-readable path to completion.

    Expected behavior: no ``SystemExit`` is raised, stdout is non-empty
    (the table renderer ran), and stderr is empty.
    """
    _patch_search_index_for_stats(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rvl", "stats", "-i", "test-idx"])
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
