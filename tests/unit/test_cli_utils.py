from argparse import ArgumentParser, Namespace

from redisvl.cli.utils import add_index_parsing_options, create_redis_url


def _args(**overrides) -> Namespace:
    values = {
        "url": None,
        "host": "localhost",
        "port": 6379,
        "user": "default",
        "password": "",
        "ssl": False,
        "_argv": (),
    }
    values.update(overrides)
    return Namespace(**values)


def test_create_redis_url_prefers_explicit_url(monkeypatch):
    """Use the explicit Redis URL before any other connection source."""
    monkeypatch.setenv("REDIS_URL", "redis://env:6379")

    assert (
        create_redis_url(_args(url="redis://explicit:6380")) == "redis://explicit:6380"
    )


def test_create_redis_url_prefers_explicit_connection_flags_over_env(monkeypatch):
    """Use explicit host and port flags before the REDIS_URL environment variable."""
    monkeypatch.setenv("REDIS_URL", "redis://env:6379")

    assert (
        create_redis_url(
            _args(
                host="cache.local",
                port=6380,
                _argv=("--host", "cache.local", "-p", "6380"),
            )
        )
        == "redis://cache.local:6380"
    )


def test_create_redis_url_uses_env_when_no_cli_connection_options(monkeypatch):
    """Use REDIS_URL when no explicit CLI connection options are provided."""
    monkeypatch.setenv("REDIS_URL", "redis://env:6379")

    assert create_redis_url(_args()) == "redis://env:6379"


def test_create_redis_url_falls_back_to_local_default(monkeypatch):
    """Fall back to the local Redis default when no other connection source is set."""
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert create_redis_url(_args()) == "redis://localhost:6379"


def test_create_redis_url_builds_ssl_url_without_double_scheme(monkeypatch):
    """Build a valid rediss URL when SSL is enabled."""
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert (
        create_redis_url(
            _args(
                host="cache.local",
                port=6380,
                user="alice",
                password="secret",
                ssl=True,
                _argv=(
                    "--host",
                    "cache.local",
                    "-p",
                    "6380",
                    "--user",
                    "alice",
                    "-a",
                    "secret",
                    "--ssl",
                ),
            )
        )
        == "rediss://alice:secret@cache.local:6380"
    )


def test_create_redis_url_omits_default_username_for_local_connections(monkeypatch):
    """Omit implicit auth when building the default local Redis URL."""
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert (
        create_redis_url(_args(host="localhost", port=6379)) == "redis://localhost:6379"
    )


def test_create_redis_url_supports_password_only_auth(monkeypatch):
    """Allow password-only Redis auth without injecting a username."""
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert (
        create_redis_url(
            _args(
                host="cache.local",
                password="secret",
                _argv=("--host", "cache.local", "-a", "secret"),
            )
        )
        == "redis://:secret@cache.local:6379"
    )


def test_parser_defaults_do_not_override_env(monkeypatch):
    """Preserve parser defaults without letting them outrank REDIS_URL."""
    parser = add_index_parsing_options(ArgumentParser())
    monkeypatch.setenv("REDIS_URL", "redis://env:6379")

    args = parser.parse_args([])
    args._argv = ()

    assert args.host == "localhost"
    assert args.port == 6379
    assert create_redis_url(args) == "redis://env:6379"


def test_explicit_default_host_still_overrides_env(monkeypatch):
    """Treat an explicitly provided default host as higher priority than REDIS_URL."""
    parser = add_index_parsing_options(ArgumentParser())
    monkeypatch.setenv("REDIS_URL", "redis://env:6379")

    argv = ["--host", "localhost"]
    args = parser.parse_args(argv)
    args._argv = tuple(argv)

    assert create_redis_url(args) == "redis://localhost:6379"
