import json
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from redisvl.cli.index import Index, _index_info_for_json
from redisvl.exceptions import RedisSearchError
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


def test_rvl_index_subprocess_help():
    """Tests that ``rvl index --help`` works via the runner module.

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


@pytest.mark.parametrize("command", ["listall", "info"])
def test_index_human_output(monkeypatch, capsys, redis_url, cli_index, command: str):
    """Tests that human-output index commands print content.

    Expected behavior: no ``SystemExit`` is raised, stdout is non-empty, and
    stderr is empty.
    """
    if command == "listall":
        argv = ["rvl", "index", "listall", "--url", redis_url]
    else:
        argv = [
            "rvl",
            "index",
            "info",
            "-i",
            cli_index.schema.index.name,
            "--url",
            redis_url,
        ]
    monkeypatch.setattr(sys, "argv", argv)

    Index()
    captured = capsys.readouterr()

    assert captured.out != ""
    assert captured.err == ""


@pytest.mark.parametrize(
    ("argv", "expected_stderr_fragments"),
    [
        # Base ``rvl index`` usage errors.
        pytest.param(["rvl", "index"], (), id="missing-subcommand"),
        pytest.param(["rvl", "index", "--json"], (), id="missing-subcommand-json"),
        pytest.param(
            ["rvl", "index", "notacommand"],
            _INDEX_SUBCOMMANDS,
            id="unknown-subcommand",
        ),
        # ``create`` input errors.
        pytest.param(["rvl", "index", "create"], (), id="create-missing-schema"),
        pytest.param(
            ["rvl", "index", "create", "-s", "/does/not/exist.yaml"],
            (),
            id="create-schema-input-error",
        ),
        # ``info`` target selection errors.
        pytest.param(["rvl", "index", "info"], (), id="info-missing-target"),
        pytest.param(
            ["rvl", "index", "info", "--json"], (), id="info-missing-target-json"
        ),
    ],
)
def test_index_usage_errors(
    monkeypatch,
    capsys,
    argv: list[str],
    expected_stderr_fragments: tuple[str, ...],
):
    """Tests that usage/input errors are reported consistently.

    Expected behavior: ``SystemExit`` code is 2, stdout is empty, stderr is
    non-empty, and selected cases include expected help fragments.
    """
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert captured.out == ""
    assert captured.err != ""
    for fragment in expected_stderr_fragments:
        assert fragment in captured.err


@pytest.mark.parametrize(
    ("argv", "patch_target", "error"),
    [
        # ``create`` Redis failures.
        pytest.param(
            ["rvl", "index", "create", "-s", "fake.yaml"],
            "create",
            RedisSearchError("create failed"),
            id="create-redis-search-error",
        ),
        # ``listall`` generic runtime failures.
        pytest.param(
            ["rvl", "index", "listall"],
            "listall",
            RuntimeError("redis unavailable"),
            id="listall",
        ),
        pytest.param(
            ["rvl", "index", "listall", "--json"],
            "listall",
            RuntimeError("redis unavailable"),
            id="listall-json",
        ),
        # ``info`` generic runtime failures.
        pytest.param(
            ["rvl", "index", "info", "-i", "test-idx"],
            "info",
            RuntimeError("boom"),
            id="info-runtime",
        ),
        pytest.param(
            ["rvl", "index", "info", "-i", "test-idx", "--json"],
            "info",
            RuntimeError("boom"),
            id="info-runtime-json",
        ),
        # ``info`` Redis search failures.
        pytest.param(
            ["rvl", "index", "info", "-i", "test-idx"],
            "info",
            RedisSearchError("Unknown index name"),
            id="info-missing-index",
        ),
        pytest.param(
            ["rvl", "index", "info", "-i", "test-idx", "--json"],
            "info",
            RedisSearchError("Unknown index name"),
            id="info-missing-index-json",
        ),
    ],
)
def test_index_runtime_errors(
    monkeypatch,
    capsys,
    argv: list[str],
    patch_target: str,
    error: BaseException,
):
    """Tests that runtime/Redis failures are reported consistently.

    Expected behavior: ``SystemExit`` code is 1, stdout is empty, and stderr
    is non-empty.
    """
    if patch_target == "create":
        _patch_search_index(
            monkeypatch,
            create_error=error,
            index_name="test-idx",
        )
    elif patch_target == "listall":
        monkeypatch.setattr(
            "redisvl.cli.index.RedisConnectionFactory.get_redis_connection",
            MagicMock(side_effect=error),
        )
    elif patch_target == "info":
        _patch_search_index_for_info(monkeypatch, info_error=error)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        Index()
    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err != ""
