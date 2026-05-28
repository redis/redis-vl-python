import json
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from redisvl.cli.index import Index, _index_info_for_json
from redisvl.index import SearchIndex

_INDEX_SUBCOMMANDS = ("info", "create", "delete", "destroy", "listall")


def _assert_index_help_contract(help_text: str) -> None:
    """Assert that ``rvl index`` help lists every supported subcommand."""
    for name in _INDEX_SUBCOMMANDS:
        # Each supported index subcommand appears on its own help line.
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
    is non-empty. The ``--json`` case is parametrized so machine consumers
    never see partial JSON when the subcommand is missing.
    """
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Argparse usage-error exit code for missing required positional.
    assert exc_info.value.code == 2
    # No stdout output - critical so --json consumers do not see partial JSON either.
    assert captured.out == ""
    # Argparse emits a usage error on stderr
    assert captured.err != ""


def test_rvl_index_unknown_subcommand(monkeypatch, capsys):
    """Tests that ``rvl index <unknown>`` reports the bad token via argparse.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    lists every valid subcommands.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "notacommand"])

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Documented usage-error exit for an unknown subcommand.
    assert exc_info.value.code == 2
    # No normal output for the invalid subcommand path.
    assert captured.out == ""
    for name in _INDEX_SUBCOMMANDS:
        # Stderr lists every valid subcommand.
        assert name in captured.err


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


@pytest.fixture
def cli_index(redis_url, redis_test_name):
    """Create a real temporary Redis index for CLI commands that inspect Redis state."""
    index_name = redis_test_name("cli_index")
    prefix = redis_test_name("cli_doc")
    index = SearchIndex.from_dict(
        {
            "index": {
                "name": index_name,
                "prefix": prefix,
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


def _patch_search_index(
    monkeypatch,
    *,
    create_error: BaseException | None = None,
    from_yaml_error: BaseException | None = None,
    index_name: str = "test-idx",
) -> None:
    """Patch ``SearchIndex.from_yaml`` for ``rvl index create`` tests."""
    fake_index = SimpleNamespace(
        schema=SimpleNamespace(index=SimpleNamespace(name=index_name)),
        create=MagicMock(side_effect=create_error),
    )
    fake_search_index = SimpleNamespace(
        from_yaml=MagicMock(side_effect=from_yaml_error, return_value=fake_index)
    )
    monkeypatch.setattr("redisvl.cli.index.SearchIndex", fake_search_index)


def _patch_search_index_for_info(
    monkeypatch,
    *,
    info_error: BaseException,
    index_name: str = "test-idx",
) -> None:
    """Patch ``SearchIndex(...)`` to exercise ``rvl index info`` failure paths."""
    fake_index = SimpleNamespace(
        schema=SimpleNamespace(index=SimpleNamespace(name=index_name)),
        info=MagicMock(side_effect=info_error),
    )
    monkeypatch.setattr(
        "redisvl.cli.index.SearchIndex", MagicMock(return_value=fake_index)
    )


def test_create_missing_schema(monkeypatch, capsys):
    """Tests that ``rvl index create`` without ``-s`` exits with a usage error.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is non-empty.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "create"])

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Documented usage-error exit when -s is not provided.
    assert exc_info.value.code == 2
    # Nothing leaks on this error path.
    assert captured.out == ""
    # Some explanatory message reaches stderr.
    assert captured.err != ""


def test_create_schema_input_error(monkeypatch, capsys):
    """Tests that ``rvl index create`` reports schema-load failures on stderr.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is exactly the raised exception's message followed by a newline.
    """
    schema_error_message = "schema file missing: /does/not/exist.yaml"

    _patch_search_index(
        monkeypatch, from_yaml_error=FileNotFoundError(schema_error_message)
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

    Expected behavior: ``SystemExit`` code is 1, stdout is empty, and the
    underlying error message reaches stderr.
    """
    from redisvl.exceptions import RedisSearchError

    redis_error_message = "create failed"

    _patch_search_index(
        monkeypatch,
        create_error=RedisSearchError(redis_error_message),
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
    # The underlying error message reaches stderr so the user knows what failed.
    assert redis_error_message in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "index", "create", "-s", "fake.yaml"],
        ["rvl", "index", "create", "-s", "fake.yaml", "--json"],
    ],
)
def test_create_success(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl index create`` succeeds with the documented banner.

    Expected behavior: no ``SystemExit`` is raised, stdout is exactly
    ``Index created successfully\\n``, and stderr is empty. ``--json`` is
    parametrized to confirm it does not invent a JSON contract for ``create``.
    """
    _patch_search_index(monkeypatch)
    monkeypatch.setattr(sys, "argv", argv)

    # Success path must return cleanly, not raise SystemExit.
    Index()
    captured = capsys.readouterr()

    # Exact stdout: the success banner with print()'s trailing newline and nothing else.
    assert captured.out == "Index created successfully\n"
    # Success path stays clean on stderr.
    assert captured.err == ""


def test_listall_json(monkeypatch, capsys, redis_url, cli_index):
    """Tests that ``rvl index listall --json`` prints the documented JSON contract.

    Expected behavior: no ``SystemExit`` is raised, stdout is one JSON line
    of the form ``{"indices": [...]}`` (no human banner), and stderr is empty.
    """
    monkeypatch.setattr(
        sys, "argv", ["rvl", "index", "listall", "--json", "--url", redis_url]
    )

    Index()
    captured = capsys.readouterr()
    out = captured.out.strip()

    # Single machine-readable line, nothing else on stdout.
    assert out.count("\n") == 0
    payload = json.loads(out)
    # Stable top-level JSON contract: only the "indices" key.
    assert list(payload) == ["indices"]
    # The command reflects the real Redis index created by the test fixture.
    assert cli_index.schema.index.name in payload["indices"]
    # JSON success path is silent on stderr.
    assert captured.err == ""


def test_listall_table(monkeypatch, capsys, redis_url, cli_index):
    """Tests that default ``rvl index listall`` prints the human-readable table.

    Expected behavior: stdout contains the test index,
    and stderr is empty.
    """
    monkeypatch.setattr(sys, "argv", ["rvl", "index", "listall", "--url", redis_url])
    Index()
    captured = capsys.readouterr()

    # The table output includes the Redis index.
    assert cli_index.schema.index.name in captured.out
    # Table success is silent on stderr.
    assert captured.err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "index", "listall"],
        ["rvl", "index", "listall", "--json"],
    ],
)
def test_listall_runtime_error(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl index listall`` reports runtime failures on stderr.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty (no
    half-formed output), and stderr is non-empty. ``--json`` is parametrized
    to confirm the contract is uniform.
    """
    monkeypatch.setattr(
        "redisvl.cli.index.RedisConnectionFactory.get_redis_connection",
        MagicMock(side_effect=RuntimeError("redis unavailable")),
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Generic runtime failure exits 1 from __init__'s catch-all.
    assert exc_info.value.code == 1
    # Failure happens before any rendering - nothing on stdout.
    assert captured.out == ""
    # The failure is surfaced on stderr.
    assert captured.err != ""


def test_info_json_normalize():
    """Tests that ``_index_info_for_json`` maps FT.INFO lists to structured JSON.

    Expected behavior: the input dict is not mutated and the returned payload
    is exactly the documented ``index_information`` + ``index_fields`` shape.
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


def test_info_json(monkeypatch, capsys, redis_url, cli_index):
    """Tests that ``rvl index info --json`` prints the documented JSON contract.

    Expected behavior: no ``SystemExit`` is raised, stdout is one parseable
    JSON line with the documented top-level sections (no table banners),
    and stderr is empty.
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rvl",
            "index",
            "info",
            "-i",
            cli_index.schema.index.name,
            "--json",
            "--url",
            redis_url,
        ],
    )

    Index()
    captured = capsys.readouterr()
    out = captured.out.strip()

    # Single line for machine consumers.
    assert out.count("\n") == 0
    payload = json.loads(out)
    # Top-level sections are stable and ordered.
    assert list(payload) == ["index_information", "index_fields"]
    # Summary section is derived from the real Redis index.
    assert payload["index_information"]["index_name"] == cli_index.schema.index.name
    assert (
        payload["index_information"]["storage_type"]
        == cli_index.schema.index.storage_type.value.upper()
    )
    assert cli_index.schema.index.prefix in payload["index_information"]["prefixes"]
    # One normalized field row from the real schema. Redis may include default
    # field options such as TEXT weight, so assert the stable identity fields.
    assert len(payload["index_fields"]) == 1
    assert {
        key: payload["index_fields"][0][key] for key in ("name", "attribute", "type")
    } == {"name": "body", "attribute": "body", "type": "TEXT"}
    # JSON success path is silent on stderr.
    assert captured.err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "index", "info", "-i", "test-idx"],
        ["rvl", "index", "info", "-i", "test-idx", "--json"],
    ],
)
def test_info_runtime_error(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl index info`` reports runtime failures on stderr.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty (no
    half-formed output), and stderr is non-empty. ``--json`` is parametrized
    to confirm the contract is uniform.
    """
    _patch_search_index_for_info(monkeypatch, info_error=RuntimeError("boom"))
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Generic runtime failure exits 1 from __init__'s catch-all.
    assert exc_info.value.code == 1
    # Failure happens before any rendering - nothing on stdout.
    assert captured.out == ""
    # The failure is surfaced on stderr.
    assert captured.err != ""


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "index", "info"],
        ["rvl", "index", "info", "--json"],
    ],
)
def test_info_no_target(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl index info`` without ``-i`` or ``-s`` exits with a usage error.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, and stderr
    is non-empty. ``--json`` is parametrized to confirm no JSON contract is
    invented for usage errors.
    """
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # _connect_to_index uses argparse-style usage exit code
    assert exc_info.value.code == 2
    # Usage errors must not pollute stdout, even with --json.
    assert captured.out == ""
    # Some explanatory message reaches stderr; exact wording is not part of the contract.
    assert captured.err != ""


def test_info_table(monkeypatch, capsys, redis_url, cli_index):
    """Tests that default ``rvl index info`` runs to completion on the table path.

    Expected behavior: no ``SystemExit`` is raised, stdout is non-empty, and
    stderr is empty. Exact rendering is a tabulate-library detail; data
    correctness is pinned by ``test_info_json``.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rvl",
            "index",
            "info",
            "-i",
            cli_index.schema.index.name,
            "--url",
            redis_url,
        ],
    )

    Index()
    captured = capsys.readouterr()

    # Table path produces some output - the renderer ran and printed cells.
    assert captured.out != ""
    # Table success is silent on stderr.
    assert captured.err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["rvl", "index", "info", "-i", "test-idx"],
        ["rvl", "index", "info", "-i", "test-idx", "--json"],
    ],
)
def test_info_missing_index(monkeypatch, capsys, argv: list[str]):
    """Tests that ``rvl index info -i <missing>`` reports the failure on stderr.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty, and the
    underlying error message reaches stderr. ``--json`` is parametrized to
    confirm the contract does not change.
    """
    from redisvl.exceptions import RedisSearchError

    underlying_error = "Unknown index name"

    _patch_search_index_for_info(
        monkeypatch, info_error=RedisSearchError(underlying_error)
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    # Documented exit code for a Redis search-side failure.
    assert exc_info.value.code == 1
    # Nothing leaks on this error path.
    assert captured.out == ""
    # The underlying Redis error reaches stderr.
    assert underlying_error in captured.err
