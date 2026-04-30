import json
from argparse import ArgumentParser
from typing import Optional

import pytest

from redisvl.cli.utils import (
    add_index_parsing_options,
    add_json_output_flag,
    cli_print_json,
    create_redis_url,
)


@pytest.fixture
def parse_args():
    parser = add_index_parsing_options(ArgumentParser())

    def _parse(argv: list[str]):
        return parser.parse_args(argv)

    return _parse


def test_parser_leaves_connection_options_unset_by_default(parse_args):
    """Leave connection options unset so URL resolution can apply precedence."""
    args = parse_args([])

    assert args.host is None
    assert args.port is None
    assert args.user is None
    assert args.password is None
    assert args.ssl is False


@pytest.mark.parametrize(
    ("argv", "env_url", "expected"),
    [
        pytest.param(
            ["--url=redis://explicit:6380"],
            "redis://env:6379",
            "redis://explicit:6380",
            id="explicit-url",
        ),
        pytest.param(
            ["--host=cache.local", "--port=6380"],
            "redis://env:6379",
            "redis://cache.local:6380",
            id="explicit-host-port",
        ),
        pytest.param(
            ["--host=cache.local", "--user=alice", "--password=secret", "--ssl"],
            None,
            "rediss://alice:secret@cache.local:6379",
            id="ssl-with-auth",
        ),
        pytest.param(
            ["--host=cache.local", "--user=alice", "--password=p@ss:w/rd?#"],
            None,
            "redis://alice:p%40ss%3Aw%2Frd%3F%23@cache.local:6379",
            id="encodes-reserved-auth-characters",
        ),
        pytest.param(
            ["--ssl"],
            "redis://production-host:6380/0",
            "rediss://production-host:6380/0",
            id="ssl-modifies-environment-url",
        ),
        pytest.param(
            ["--host=cache.local", "-a", "secret"],
            None,
            "redis://:secret@cache.local:6379",
            id="password-only-auth",
        ),
        pytest.param(
            ["--user="],
            "redis://env:6379",
            "redis://env:6379",
            id="empty-user-does-not-override-env",
        ),
        pytest.param(
            ["--host=localhost"],
            "redis://env:6379",
            "redis://localhost:6379",
            id="explicit-default-host",
        ),
        pytest.param(
            [],
            "redis://env:6379",
            "redis://env:6379",
            id="environment-fallback",
        ),
        pytest.param(
            [],
            None,
            "redis://localhost:6379",
            id="local-default",
        ),
    ],
)
def test_create_redis_url_resolves_connection_sources(
    parse_args, monkeypatch, argv: list[str], env_url: Optional[str], expected: str
):
    """Resolve Redis URLs from CLI args, environment, and local defaults."""
    if env_url is None:
        monkeypatch.delenv("REDIS_URL", raising=False)
    else:
        monkeypatch.setenv("REDIS_URL", env_url)

    assert create_redis_url(parse_args(argv)) == expected


def test_add_json_output_flag_absent_is_false():
    """``add_json_output_flag`` leaves ``args.json`` false when ``--json`` is omitted.

    Expected: callers can treat the default as non-JSON (human-oriented) output.
    """
    parser = add_json_output_flag(ArgumentParser())
    args = parser.parse_args([])
    assert args.json is False


def test_add_json_output_flag_present_is_true():
    """``add_json_output_flag`` sets ``args.json`` true when ``--json`` is passed.

    Expected: parsed args reflect machine-readable JSON mode.
    """
    parser = add_json_output_flag(ArgumentParser())
    args = parser.parse_args(["--json"])
    assert args.json is True


def test_cli_print_json_writes_single_json_object(capsys):
    """``cli_print_json`` writes exactly one JSON object to stdout for a string-only dict.

    Expected: stdout is a single line parseable by ``json.loads``, round-tripping the
    payload, with no extra newlines beyond what ``print`` adds for one line.
    """
    payload = {"version": "0.0.0"}
    cli_print_json(payload)
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == payload
    assert out.count("\n") == 0


def test_cli_print_json_encodes_bytes_values(capsys):
    """``cli_print_json`` serializes ``bytes`` values via the JSON default handler.

    Invalid UTF-8 byte sequences are replaced (U+FFFD) in the decoded string, matching
    ``_cli_json_default`` so future Redis-centric payloads can be emitted safely.
    """
    cli_print_json({"blob": b"abc\xff"})
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {"blob": "abc\ufffd"}
