"""Portable index configurations must not carry credentials or open clients."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
import yaml

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.schema import IndexSchema


@pytest.fixture(params=[SearchIndex, AsyncSearchIndex])
def index_class(request):
    return request.param


@pytest.fixture
def schema():
    return {
        "index": {"name": "serialized", "prefix": "documents"},
        "fields": [
            {"name": "text", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {"algorithm": "flat", "dims": 3, "distance_metric": "cosine"},
            },
        ],
    }


@pytest.fixture(autouse=True)
def no_connections(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Serialization must not create or validate a Redis connection")

    for name in (
        "get_redis_connection",
        "get_async_redis_connection",
        "_get_aredis_connection",
        "validate_sync_redis",
        "validate_async_redis",
    ):
        monkeypatch.setattr(RedisConnectionFactory, name, fail)


def test_default_export_is_schema_only(index_class, schema, tmp_path):
    index = index_class.from_dict(schema, redis_url="redis://user:secret@host:6379")
    config = index.to_dict()
    assert config == index.schema.to_dict()
    assert index_class.from_dict(config).schema == index.schema
    path = tmp_path / "schema.yaml"
    index.to_yaml(path)
    assert yaml.safe_load(path.read_text()) == config
    assert index_class.from_yaml(path).schema == index.schema
    assert "secret" not in path.read_text()


def test_json_schema_roundtrip(index_class, schema):
    schema["index"]["storage_type"] = "json"
    schema["fields"][0]["path"] = "$.text"
    schema["fields"][1]["path"] = "$.embedding"
    index = index_class.from_dict(schema)
    assert index_class.from_dict(index.to_dict()).schema == index.schema


@pytest.mark.parametrize(
    "url",
    [
        "redis://localhost:6379/2",
        "rediss://[2001:db8::1]:6380/3",
        "unix:///tmp/redis.sock",
        "redis+sentinel://host1:26379,host2:26380/my%2Dmaster/1",
    ],
)
def test_connection_roundtrip(index_class, schema, tmp_path, url):
    options = {
        "socket_timeout": 2.5,
        "decode_responses": True,
        "ssl_cert_reqs": "required",
    }
    index = index_class.from_dict(schema, redis_url=url, connection_kwargs=options)
    config = index.to_dict(include_connection=True)
    original = deepcopy(config)
    restored = index_class.from_dict(config)
    assert restored._redis_url == url
    assert restored._connection_kwargs == options
    assert restored.schema == index.schema
    assert config == original
    path = tmp_path / "index.yaml"
    index.to_yaml(path, include_connection=True)
    assert index_class.from_yaml(path).to_dict(include_connection=True) == config
    restored._connection_kwargs["socket_timeout"] = 10
    assert config == original
    assert options["socket_timeout"] == 2.5


@pytest.mark.parametrize(
    "url,endpoint",
    [
        ("redis://alice:p%40ss@[::1]:6379/0", "redis://[::1]:6379/0"),
        ("redis://alice:@host/0", "redis://host/0"),
        ("redis://alice@host/0", "redis://host/0"),
        ("redis://host/0?password=p%40ss&db=2", "redis://host/0"),
        ("rediss://host/0?ssl_password=p%40ss", "rediss://host/0"),
        ("redis://host/0#p%40ss", "redis://host/0"),
        (
            "redis+sentinel://alice:p%40ss@a:26379,b:26379/master/1",
            "redis+sentinel://a:26379,b:26379/master/1",
        ),
        ("unix://alice:p%40ss@/tmp/redis.sock?db=2", "unix:///tmp/redis.sock"),
    ],
)
def test_redaction_requires_explicit_restore(
    index_class, schema, tmp_path, monkeypatch, url, endpoint
):
    options = {
        "password": "kw-secret",
        "credential_provider": Mock(),
        "socket_timeout": 1,
    }
    index = index_class.from_dict(schema, redis_url=url, connection_kwargs=options)
    config = index.to_dict(include_connection=True)
    assert config["connection"] == {
        "redis_url": endpoint,
        "connection_kwargs": {"socket_timeout": 1},
        "requires_override": True,
    }
    path = tmp_path / "safe.yaml"
    index.to_yaml(path, include_connection=True)
    for secret in ("alice", "p%40ss", "kw-secret", "credential_provider"):
        assert secret not in json.dumps(config)
        assert secret not in path.read_text()
    monkeypatch.setenv("REDIS_URL", "redis://wrong-host:6379")
    with pytest.raises(ValueError, match="explicit redis_url or redis_client"):
        index_class.from_dict(config)
    with pytest.raises(ValueError, match="explicit redis_url or redis_client"):
        index_class.from_yaml(path)
    restored = index_class.from_dict(config, redis_url=url, connection_kwargs=options)
    assert restored._redis_url == url
    assert restored._connection_kwargs == options


@pytest.mark.parametrize(
    "override",
    [
        {"redis_url": None},
        {"redis_url": ""},
        {"connection_kwargs": {"password": "new"}},
    ],
)
def test_incomplete_override_does_not_fall_back(index_class, schema, override):
    index = index_class.from_dict(schema, redis_url="redis://user:secret@host")
    with pytest.raises(ValueError, match="explicit redis_url or redis_client"):
        index_class.from_dict(index.to_dict(include_connection=True), **override)


def test_explicit_endpoint_does_not_inherit_connection_options(index_class, schema):
    index = index_class.from_dict(
        schema, redis_url="rediss://original", connection_kwargs={"ssl": True}
    )
    config = index.to_dict(include_connection=True)
    restored = index_class.from_dict(config, redis_url="redis://other")
    assert restored._redis_url == "redis://other"
    assert restored._connection_kwargs == {}
    client = Mock()
    restored = index_class.from_dict(config, redis_client=client)
    assert restored._client_was_injected
    assert restored._redis_url is None
    assert restored._connection_kwargs == {}


def test_explicit_connection_kwargs_replace_stored_options(index_class, schema):
    index = index_class.from_dict(
        schema, redis_url="redis://host", connection_kwargs={"socket_timeout": 1}
    )
    restored = index_class.from_dict(
        index.to_dict(include_connection=True), connection_kwargs={"protocol": 3}
    )
    assert restored._redis_url == "redis://host"
    assert restored._connection_kwargs == {"protocol": 3}


@pytest.mark.parametrize("injected", [False, True])
def test_unavailable_connection_metadata_requires_replacement(
    index_class, schema, monkeypatch, injected
):
    monkeypatch.setenv("REDIS_URL", "redis://env-user:env-secret@host")
    client = Mock() if injected else None
    index = index_class.from_dict(
        schema, redis_client=client, redis_url="redis://unused" if injected else None
    )
    config = index.to_dict(include_connection=True)
    assert config["connection"]["redis_url"] is None
    assert config["connection"]["requires_override"] is True
    assert "env-secret" not in json.dumps(config)
    if client is not None:
        assert client.mock_calls == []
    with pytest.raises(ValueError, match="explicit redis_url or redis_client"):
        index_class.from_dict(config)
    assert index_class.from_dict(config, redis_client=Mock())._client_was_injected


@pytest.mark.parametrize(
    "options",
    [
        {"ssl_cert_reqs": "secret"},
        {"socket_timeout": "secret"},
        {"decode_responses": Mock()},
        {"sentinel_kwargs": {"password": "secret"}},
    ],
)
def test_allowlist_rejects_unsafe_values(index_class, schema, options):
    index = index_class.from_dict(
        schema, redis_url="redis://host", connection_kwargs=options
    )
    config = index.to_dict(include_connection=True)
    assert config["connection"]["connection_kwargs"] == {}
    assert config["connection"]["requires_override"] is True
    assert "secret" not in json.dumps(config)


@pytest.mark.parametrize(
    "url",
    ["rediss://[invalid-secret", "http://user:secret@host", "redis:///missing-host"],
)
def test_invalid_url_errors_do_not_expose_input(index_class, schema, url):
    index = index_class.from_dict(schema, redis_url=url)
    with pytest.raises(ValueError) as error:
        index.to_dict(include_connection=True)
    assert url not in str(error.value)
    assert "secret" not in str(error.value)


def test_export_error_preserves_existing_file(index_class, schema, tmp_path):
    path = tmp_path / "existing.yaml"
    path.write_text("existing configuration")
    index = index_class.from_dict(schema, redis_url="rediss://[invalid")
    with pytest.raises(ValueError):
        index.to_yaml(path, include_connection=True)
    assert path.read_text() == "existing configuration"


def test_yaml_overwrite_and_invalid_inputs(index_class, schema, tmp_path):
    index = index_class.from_dict(schema)
    path = tmp_path / "config.yaml"
    index.to_yaml(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        index.to_yaml(path, overwrite=False)
    assert path.read_bytes() == original
    index.to_yaml(path, overwrite=True)
    with pytest.raises(FileNotFoundError, match="Schema file .* does not exist"):
        index_class.from_yaml(tmp_path / "missing.yaml")
    path.write_text("[")
    with pytest.raises(yaml.YAMLError):
        index_class.from_yaml(path)
    path.write_text("")
    # Keep the existing schema loader's error for an empty YAML document.
    with pytest.raises(AttributeError) as schema_error:
        IndexSchema.from_yaml(path)
    with pytest.raises(type(schema_error.value)):
        index_class.from_yaml(path)


@pytest.mark.parametrize(
    "connection",
    [
        None,
        [],
        {"redis_url": 12},
        {"redis_url": "redis://host", "requires_override": "false"},
        {"redis_url": "redis://host", "connection_kwargs": []},
    ],
)
def test_invalid_connection_metadata(index_class, schema, connection):
    with pytest.raises(ValueError, match="connection"):
        index_class.from_dict({**schema, "connection": connection})
