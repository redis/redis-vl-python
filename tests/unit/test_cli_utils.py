from argparse import Namespace

from redisvl.cli.utils import create_redis_url


def _args(**overrides) -> Namespace:
    values = {
        "url": None,
        "host": None,
        "port": None,
        "user": None,
        "password": None,
        "ssl": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_create_redis_url_prefers_explicit_url(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://env:6379")

    assert (
        create_redis_url(_args(url="redis://explicit:6380")) == "redis://explicit:6380"
    )


def test_create_redis_url_prefers_explicit_connection_flags_over_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://env:6379")

    assert (
        create_redis_url(_args(host="cache.local", port=6380))
        == "redis://cache.local:6380"
    )


def test_create_redis_url_uses_env_when_no_cli_connection_options(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://env:6379")

    assert create_redis_url(_args()) == "redis://env:6379"


def test_create_redis_url_falls_back_to_local_default(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert create_redis_url(_args()) == "redis://localhost:6379"


def test_create_redis_url_builds_ssl_url_without_double_scheme(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert (
        create_redis_url(
            _args(
                host="cache.local",
                port=6380,
                user="alice",
                password="secret",
                ssl=True,
            )
        )
        == "rediss://alice:secret@cache.local:6380"
    )


def test_create_redis_url_omits_default_username_for_local_connections(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert (
        create_redis_url(_args(host="localhost", port=6379)) == "redis://localhost:6379"
    )


def test_create_redis_url_supports_password_only_auth(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert (
        create_redis_url(_args(host="cache.local", password="secret"))
        == "redis://:secret@cache.local:6379"
    )
