"""Unit tests for SearchIndex._to_async. No server is needed: nothing connects."""

import pytest
from redis import Redis

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema

URL = "redis://h:6379"
SCHEMA = {
    "index": {"name": "idx", "prefix": "idx"},
    "fields": [{"name": "t", "type": "tag"}],
}


@pytest.mark.parametrize("source", ["url", "env", "client"])
def test_to_async_builds_an_owned_copy_from_the_same_address(source, monkeypatch):
    # A Sentinel address cannot be converted, so the env case must not try.
    monkeypatch.setenv("REDIS_URL", "redis+sentinel://h:26379/master")
    # Reading an injected client reports lib_name to the server; skip that.
    monkeypatch.setattr("redisvl.redis.connection._identify_client", lambda *a: None)
    connection = {
        "url": {"redis_url": URL},
        "env": {},
        "client": {"redis_client": Redis(host="h")},
    }[source]
    index = SearchIndex(
        IndexSchema.from_dict(SCHEMA),
        validate_on_load=True,
        lib_name="wrapper",
        **connection,
    )

    aindex = index._to_async()

    assert aindex._redis_url == (URL if source == "url" else None)
    # Only an injected client is converted; otherwise the copy connects lazily.
    assert (aindex._redis_client is not None) == (source == "client")
    assert aindex._owns_redis_client and aindex._validate_on_load
    assert aindex._lib_name == "wrapper"
    assert aindex.schema is index.schema
