from argparse import ArgumentParser
from typing import Optional

import pytest

from redisvl.cli.utils import add_index_parsing_options, create_redis_url


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
