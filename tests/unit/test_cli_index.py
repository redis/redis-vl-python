import json
import re
import subprocess
import sys

import pytest

from redisvl.cli.index import Index, _index_info_for_json

_INDEX_SUBCOMMANDS = ("info", "create", "delete", "destroy", "listall")


def _assert_index_help_contract(help_text: str) -> None:
    """Assert key help text users rely on for ``rvl index``: usage, command header, and subcommands."""
    # Includes the short usage form for the index subcommand router.
    assert "rvl index <command> [<args>]" in help_text
    # Includes the command section header that introduces the subcommand list.
    assert "Commands:" in help_text
    for name in _INDEX_SUBCOMMANDS:
        # Includes each supported index subcommand on its own help line.
        assert re.search(rf"^\s*{re.escape(name)}\s+", help_text, re.MULTILINE)


@pytest.mark.parametrize("argv", [["rvl", "index", "--help"], ["rvl", "index", "-h"]])
def test_rvl_index_help(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl index --help`` and ``-h`` are discoverable.

    Expected behavior: ``SystemExit`` code is 0, stdout contains the documented
    ``rvl index`` help contract, and stderr is empty.
    """
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Help requests terminate successfully.
    assert exc_info.value.code == 0
    # Successful help output does not leak to stderr.
    assert captured.err == ""
    # stdout contains the documented ``rvl index`` help contract.
    _assert_index_help_contract(captured.out)


@pytest.mark.parametrize("argv", [["rvl", "index"], ["rvl", "index", "--json"]])
def test_rvl_index_no_subcommand(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl index`` without a subcommand fails with a usage error.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is argparse's usage line. The ``--json`` case is parametrized so machine
    consumers never see partial JSON when the subcommand is missing.
    """
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Argparse usage-error exit code for missing required positional.
    assert exc_info.value.code == 2
    # No stdout output - critical so --json consumers do not see partial JSON either.
    assert captured.out == ""
    # Argparse usage line is rendered to stderr.
    assert "usage:" in captured.err.lower()
    # Stderr identifies the parser by its prog name (``rvl index``).
    assert "rvl index" in captured.err


def test_rvl_index_unknown_subcommand(monkeypatch, capsys):
    """Tests that ``rvl index <unknown>`` reports the bad token and prints help.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    contains ``Unknown command: <token>`` followed by the full help text.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "notacommand"])

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Documented usage-error exit for an unknown subcommand.
    assert exc_info.value.code == 2
    # No normal output for the invalid subcommand path.
    assert captured.out == ""
    # Stderr identifies the rejected subcommand token.
    assert "Unknown command: notacommand" in captured.err
    for name in _INDEX_SUBCOMMANDS:
        # Stderr help still lists every valid index subcommand.
        assert name in captured.err
    # Stderr help includes the command section header from the help text.
    assert "Commands:" in captured.err


def test_rvl_index_subprocess_help():
    """End-to-end smoke test of ``rvl index --help`` via the runner module.

    Expected behavior: the subprocess exits 0, stdout matches the same help
    contract as the in-process test, and stderr is empty - confirming the
    installed entrypoint wires through to ``Index`` correctly.
    """
    result = subprocess.run(
        [sys.executable, "-m", "redisvl.cli.runner", "index", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    # Process exited cleanly.
    assert result.returncode == 0
    # Help is not emitted on stderr.
    assert result.stderr == ""
    # stdout includes the same help contract as the in-process help test.
    _assert_index_help_contract(result.stdout)


def _raise(exc: BaseException):
    """Return a zero-arg callable that raises ``exc``."""

    def _do():
        raise exc

    return _do


class _FakeConn:
    def __init__(self, result, boom=False):
        self._result = result
        self._boom = boom

    def execute_command(self, cmd):
        # listall must query Redis with FT._LIST
        assert cmd == "FT._LIST"
        if self._boom:
            raise RuntimeError("redis unavailable")
        return self._result


def _patch_redis_connection(monkeypatch, *, result=None, boom: bool = False) -> None:
    """Patch ``RedisConnectionFactory.get_redis_connection`` to return a ``_FakeConn``.

    ``result`` is what ``execute_command("FT._LIST")`` returns on the success
    path (defaults to ``[]``); ``boom=True`` makes ``execute_command`` raise a
    ``RuntimeError`` instead, exercising the runtime-failure branch.
    """
    fake = _FakeConn([] if result is None else result, boom=boom)
    monkeypatch.setattr(
        "redisvl.cli.index.RedisConnectionFactory.get_redis_connection",
        lambda *a, **k: fake,
    )


def _patch_search_index(
    monkeypatch,
    *,
    create_behavior=None,
    from_yaml_raises: BaseException | None = None,
    index_name: str = "test-idx",
) -> None:
    """Patch ``redisvl.cli.index.SearchIndex`` with a minimal fake for ``rvl index create`` tests.

    ``from_yaml_raises`` makes ``from_yaml`` raise that exception; ``create_behavior``
    is invoked inside ``FakeIndex.create()`` (use :func:`_raise` for failures);
    ``index_name`` is surfaced via ``index.schema.index.name``.
    """
    if from_yaml_raises is not None:

        class FakeSearchIndex:
            @classmethod
            def from_yaml(cls, *_args, **_kwargs):
                raise from_yaml_raises

    else:

        class _FakeIndexInfo:
            name = index_name

        class _FakeSchema:
            index = _FakeIndexInfo()

        class FakeIndex:
            schema = _FakeSchema()

            def create(self):
                if create_behavior is not None:
                    create_behavior()

        class FakeSearchIndex:
            @classmethod
            def from_yaml(cls, *_args, **_kwargs):
                return FakeIndex()

    monkeypatch.setattr("redisvl.cli.index.SearchIndex", FakeSearchIndex)


def _patch_search_index_for_info(
    monkeypatch,
    *,
    info_behavior=None,
    with_field_options: bool = False,
    index_name: str = "test-idx",
) -> None:
    """Patch ``redisvl.cli.index.SearchIndex`` with a minimal fake for ``rvl index info`` tests.

    By default builds a happy-path FakeIndex whose ``info()`` returns a fixed
    FT.INFO payload (with an extra ``NOSTEM=1`` attribute when
    ``with_field_options=True``). Pass ``info_behavior=_raise(SomeError())``
    to make ``info()`` raise instead. ``index_name`` is surfaced via
    ``index.schema.index.name`` so ``exit_redis_search_error`` can format its
    message verbatim.
    """
    if info_behavior is None:
        attrs = [b"identifier", b"u", b"attribute", b"u", b"type", b"TAG"]
        if with_field_options:
            attrs.extend([b"NOSTEM", b"1"])
        payload = {
            "index_name": b"test-idx",
            "index_definition": ["key_type", b"HASH", "prefixes", [b"pre"]],
            "attributes": [attrs],
        }

        def info_behavior():
            return payload

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

        def info(self):
            return info_behavior()

    monkeypatch.setattr("redisvl.cli.index.SearchIndex", FakeIndex)


def test_create_missing_schema(monkeypatch, capsys):
    """Tests that ``rvl index create`` without ``-s`` exits with the missing-schema usage error.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is exactly ``Schema must be provided to create an index\\n``.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "create"])

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Documented usage-error exit when -s is not provided.
    assert exc_info.value.code == 2
    # Nothing leaks on this error path.
    assert captured.out == ""
    # Exact stderr contract from the missing-schema guard's print() call.
    assert captured.err == "Schema must be provided to create an index\n"


def test_create_schema_input_error(monkeypatch, capsys):
    """Tests that ``rvl index create`` reports schema-load failures on stderr.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is exactly the raised exception's message followed by a newline.
    """
    schema_error_message = "schema file missing: /does/not/exist.yaml"

    _patch_search_index(
        monkeypatch, from_yaml_raises=FileNotFoundError(schema_error_message)
    )
    monkeypatch.setattr(
        sys, "argv", ["rvl", "index", "create", "-s", "/does/not/exist.yaml"]
    )

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # exit_schema_input_error uses exit code 2 when -s was provided.
    assert exc_info.value.code == 2
    # Nothing on stdout when schema input fails.
    assert captured.out == ""
    # Exact stderr contract: exit_schema_input_error does print(str(exc), file=sys.stderr).
    assert captured.err == f"{schema_error_message}\n"


def test_create_redis_search_error(monkeypatch, capsys):
    """Tests that ``rvl index create`` reports Redis-side failures on stderr.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty, and stderr
    contains both the ``Redis search operation failed for index 'test-idx'.``
    prefix and the underlying error message.
    """
    from redisvl.exceptions import RedisSearchError

    redis_error_message = "create failed"

    _patch_search_index(
        monkeypatch,
        create_behavior=_raise(RedisSearchError(redis_error_message)),
        index_name="test-idx",
    )
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "create", "-s", "fake.yaml"])

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Documented Redis-failure exit code from exit_redis_search_error.
    assert exc_info.value.code == 1
    # No partial output before the failure.
    assert captured.out == ""
    # Documented error prefix, with the index name surfacing verbatim.
    assert "Redis search operation failed for index 'test-idx'." in captured.err
    # The exception's str() is forwarded to stderr verbatim.
    assert redis_error_message in captured.err


def test_create_success(monkeypatch, capsys):
    """Tests that ``rvl index create`` succeeds with the documented banner.

    Expected behavior: no ``SystemExit`` is raised, stdout is exactly
    ``Index created successfully\\n``, and stderr is empty.
    """
    _patch_search_index(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "create", "-s", "fake.yaml"])

    # Success path must return cleanly, not raise SystemExit.
    Index()
    captured = capsys.readouterr()

    # Exact stdout: the success banner with print()'s trailing newline and nothing else.
    assert captured.out == "Index created successfully\n"
    # Success path stays clean on stderr.
    assert captured.err == ""


def test_create_json_flag_no_json(monkeypatch, capsys):
    """Tests that ``rvl index create --json`` does not invent a JSON contract.

    Expected behavior: stdout matches the no-flag success banner exactly and
    stderr is empty - ``--json`` is a no-op for ``create``.
    """
    _patch_search_index(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rvl", "index", "create", "-s", "fake.yaml", "--json"],
    )

    # Success path must return cleanly, not call sys.exit, even with --json.
    Index()
    captured = capsys.readouterr()

    # Success path stays clean on stderr even with --json.
    assert captured.err == ""

    # stdout is identical to the no-flag success path, proving --json invents no JSON contract.
    assert captured.out == "Index created successfully\n"


def test_listall_json(monkeypatch, capsys):
    """Tests that ``rvl index listall --json`` prints the documented JSON contract.

    Expected behavior: no ``SystemExit`` is raised, stdout is one JSON line
    of the form ``{"indices": [...]}`` (no human banner), and stderr is empty.
    """
    _patch_redis_connection(monkeypatch, result=[b"idx_a", b"idx_b"])
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall", "--json"])

    try:
        Index()
    except SystemExit as exc:
        # Success path must return cleanly, not call sys.exit.
        pytest.fail(f"listall --json raised SystemExit({exc.code}) on the success path")
    captured = capsys.readouterr()
    out = captured.out.strip()

    # --json must not print the human banner.
    assert "Indices:" not in out
    # Single machine-readable line, nothing else on stdout.
    assert out.count("\n") == 0
    # Stable top-level JSON contract: only the "indices" key, in FT._LIST order.
    assert json.loads(out) == {"indices": ["idx_a", "idx_b"]}
    # JSON success path is silent on stderr.
    assert captured.err == ""


def test_listall_table(monkeypatch, capsys):
    """Tests that default ``rvl index listall`` prints the human-readable table.

    Expected behavior: stdout is exactly ``Indices:\\n1. one\\n2. two\\n``
    (header + 1-indexed rows in FT._LIST order), and stderr is empty.
    """
    _patch_redis_connection(monkeypatch, result=[b"one", b"two"])
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall"])
    Index()
    captured = capsys.readouterr()

    # Exact stdout: header + numbered rows in FT._LIST order, each from a single print().
    assert captured.out == "Indices:\n1. one\n2. two\n"
    # Table success is silent on stderr.
    assert captured.err == ""


def test_listall_json_empty(monkeypatch, capsys):
    """Tests that ``rvl index listall --json`` handles an empty FT._LIST result.

    Expected behavior: stdout is one JSON line with ``{"indices": []}`` (no
    human banner), and stderr is empty.
    """
    _patch_redis_connection(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall", "--json"])

    Index()
    captured = capsys.readouterr()
    out = captured.out.strip()

    # Empty --json must not print the human banner.
    assert "Indices:" not in out
    # Single machine-readable line for the empty case too.
    assert out.count("\n") == 0
    # Empty array is a valid success payload.
    assert json.loads(out) == {"indices": []}
    # JSON success path is silent on stderr.
    assert captured.err == ""


def test_listall_json_error(monkeypatch, capsys):
    """Tests that ``rvl index listall --json`` reports runtime failures on stderr.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty (no
    half-formed JSON), and the underlying error message is on stderr.
    """
    _patch_redis_connection(monkeypatch, boom=True)
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall", "--json"])

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # generic runtime failure exits 1 from __init__'s catch-all
    assert exc_info.value.code == 1
    # failure happens before cli_print_json — nothing on stdout
    assert captured.out == ""
    # underlying error message is surfaced on stderr
    assert "redis unavailable" in captured.err


def test_info_json_normalize():
    """Tests that ``_index_info_for_json`` maps FT.INFO lists to structured JSON.

    Expected behavior: the input dict is not mutated and the output has the
    documented top-level keys ``index_information`` and ``index_fields``.
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

    # Helper does not mutate its input.
    assert str(raw) == before
    # Exact summary + fields payload, matching what the table prints semantically.
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
    }


def test_info_json(monkeypatch, capsys):
    """Tests that ``rvl index info --json`` prints the documented JSON contract.

    Expected behavior: no ``SystemExit`` is raised, stdout is one parseable
    JSON line with the documented top-level sections (no table banners),
    and stderr is empty.
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

    _patch_search_index_for_info(monkeypatch, with_field_options=True)
    monkeypatch.setattr(
        sys, "argv", ["rvl", "index", "info", "-i", "test-idx", "--json"]
    )

    try:
        Index()
    except SystemExit as exc:
        # Success path must return cleanly, not call sys.exit.
        pytest.fail(f"info --json raised SystemExit({exc.code}) on the success path")
    captured = capsys.readouterr()
    out = captured.out.strip()

    # Single line for machine consumers.
    assert out.count("\n") == 0
    payload = json.loads(out)
    # --json must not emit table banner text.
    assert "Index Information:" not in out and "Index Fields:" not in out
    # Top-level sections are stable and ordered.
    assert list(payload) == ["index_information", "index_fields"]
    # Summary section matches table-derived values.
    assert payload["index_information"] == expected_index_information
    # One normalized field row with options.
    assert payload["index_fields"] == [expected_field]
    # JSON success path is silent on stderr.
    assert captured.err == ""


def test_info_json_error(monkeypatch, capsys):
    """Tests that ``rvl index info --json`` reports runtime failures on stderr.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty (no
    half-formed JSON), and the underlying error message is on stderr.
    """
    _patch_search_index_for_info(
        monkeypatch, info_behavior=_raise(RuntimeError("boom"))
    )
    monkeypatch.setattr(
        sys, "argv", ["rvl", "index", "info", "-i", "test-idx", "--json"]
    )

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Generic runtime failure exits 1 from __init__'s catch-all.
    assert exc_info.value.code == 1
    # No partial JSON before the exception.
    assert captured.out == ""
    # Underlying error message is surfaced on stderr.
    assert "boom" in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "index", "info"],
        ["rvl", "index", "info", "--json"],
    ],
)
def test_info_no_target(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl index info`` without ``-i`` or ``-s`` exits with the usage error.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is exactly ``Index name or schema must be provided\\n``. ``--json`` is
    parametrized to confirm no JSON contract is invented for usage errors.
    """
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # _connect_to_index uses argparse-style usage exit code
    assert exc_info.value.code == 2
    # usage errors must not pollute stdout, even with --json
    assert captured.out == ""
    # exact stderr message contract from _connect_to_index
    assert captured.err == "Index name or schema must be provided\n"


def test_info_table(monkeypatch, capsys):
    """Tests that default ``rvl index info`` prints the human-readable table.

    Expected behavior: stdout includes the ``Index Information:`` and
    ``Index Fields:`` banners and surfaces the index name and storage type;
    stderr is empty.
    """
    _patch_search_index_for_info(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "info", "-i", "test-idx"])

    Index()
    captured = capsys.readouterr()

    # Table success is silent on stderr.
    assert captured.err == ""
    # Top-of-table banner is printed.
    assert "Index Information:" in captured.out
    # Fields-table banner is printed.
    assert "Index Fields:" in captured.out
    # The index name surfaces in the rendered cells.
    assert "test-idx" in captured.out
    # The storage type surfaces in the rendered cells.
    assert "HASH" in captured.out


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "index", "info", "-i", "test-idx"],
        ["rvl", "index", "info", "-i", "test-idx", "--json"],
    ],
)
def test_info_missing_index(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl index info -i <missing>`` reports the missing index on stderr.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty, and stderr
    contains both the ``Redis search operation failed for index 'test-idx'.``
    prefix and the underlying error message. ``--json`` is parametrized to
    confirm the contract does not change.
    """
    from redisvl.exceptions import RedisSearchError

    underlying_error = "Unknown index name"

    _patch_search_index_for_info(
        monkeypatch, info_behavior=_raise(RedisSearchError(underlying_error))
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Documented exit code for a Redis search-side failure.
    assert exc_info.value.code == 1
    # Nothing leaks on this error path.
    assert captured.out == ""
    # Documented error prefix, with the index name surfacing verbatim.
    assert "Redis search operation failed for index 'test-idx'." in captured.err
    # The exception's str() is forwarded to stderr verbatim.
    assert underlying_error in captured.err
