import os
import tempfile

import pytest
import yaml

from redisvl.index.index import AsyncSearchIndex, SearchIndex
from redisvl.schema.schema import IndexSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_SCHEMA = {
    "index": {
        "name": "test-index",
        "prefix": "doc",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "title", "type": "tag"},
        {"name": "body", "type": "text"},
        {
            "name": "vector",
            "type": "vector",
            "attrs": {
                "dims": 3,
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
    ],
}


def _make_index(cls=SearchIndex, **conn):
    schema = IndexSchema.from_dict(SAMPLE_SCHEMA)
    return cls(schema=schema, **conn)


# ---------------------------------------------------------------------------
# to_dict tests
# ---------------------------------------------------------------------------


class TestToDict:
    def test_schema_only(self):
        idx = _make_index()
        d = idx.to_dict()
        assert d["index"]["name"] == "test-index"
        assert d["index"]["prefix"] == "doc"
        assert len(d["fields"]) == 3
        assert "_redis_url" not in d
        assert "_connection_kwargs" not in d

    def test_include_connection_with_url(self):
        idx = _make_index(redis_url="redis://:secret@localhost:6379")
        d = idx.to_dict(include_connection=True)
        assert d["_redis_url"] == "redis://:****@localhost:6379"

    def test_include_connection_password_only_url(self):
        """Password-only URLs (:pass@host) are sanitized correctly."""
        idx = _make_index(redis_url="redis://:mysecret@localhost:6379")
        d = idx.to_dict(include_connection=True)
        assert d["_redis_url"] == "redis://:****@localhost:6379"

    def test_include_connection_user_pass_url(self):
        idx = _make_index(redis_url="redis://admin:secret@localhost:6379")
        d = idx.to_dict(include_connection=True)
        assert d["_redis_url"] == "redis://admin:****@localhost:6379"

    def test_include_connection_username_only_url(self):
        """Username-only URLs (no password) are left unchanged."""
        idx = _make_index(redis_url="redis://readonly@localhost:6379")
        d = idx.to_dict(include_connection=True)
        assert d["_redis_url"] == "redis://readonly@localhost:6379"

    def test_include_connection_no_url(self):
        """When initialized with a client, _redis_url is None — omit it."""
        idx = _make_index()
        d = idx.to_dict(include_connection=True)
        assert "_redis_url" not in d

    def test_include_connection_filters_sensitive_kwargs(self):
        idx = _make_index(
            redis_url="redis://localhost:6379",
            connection_kwargs={"password": "s3cret", "ssl_cert_reqs": "required"},
        )
        d = idx.to_dict(include_connection=True)
        # password should NOT leak
        assert "s3cret" not in d["_connection_kwargs"]
        assert d["_connection_kwargs"] == {"ssl_cert_reqs": "required"}

    def test_async_index_to_dict(self):
        idx = _make_index(cls=AsyncSearchIndex, redis_url="redis://localhost:6379")
        d = idx.to_dict(include_connection=True)
        assert d["_redis_url"] == "redis://localhost:6379"


# ---------------------------------------------------------------------------
# to_yaml tests
# ---------------------------------------------------------------------------


class TestToYaml:
    def test_writes_yaml(self):
        idx = _make_index()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            idx.to_yaml(path)
            with open(path) as f:
                data = yaml.safe_load(f)
            assert data["index"]["name"] == "test-index"
        finally:
            os.unlink(path)

    def test_overwrite_false_raises(self):
        idx = _make_index()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            idx.to_yaml(path)
            with pytest.raises(FileExistsError):
                idx.to_yaml(path, overwrite=False)
        finally:
            os.unlink(path)

    def test_overwrite_true(self):
        idx = _make_index()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            idx.to_yaml(path)
            idx.to_yaml(path, overwrite=True)  # should not raise
        finally:
            os.unlink(path)

    def test_async_to_yaml(self):
        idx = _make_index(cls=AsyncSearchIndex)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            idx.to_yaml(path)
            with open(path) as f:
                data = yaml.safe_load(f)
            assert data["index"]["name"] == "test-index"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_roundtrip_dict_schema_only(self):
        idx = _make_index()
        d = idx.to_dict()
        restored = SearchIndex.from_dict(d)
        assert restored.schema.index.name == "test-index"

    def test_roundtrip_dict_with_connection(self):
        idx = _make_index(redis_url="redis://localhost:6379")
        d = idx.to_dict(include_connection=True)
        restored = SearchIndex.from_dict(d)
        assert restored._redis_url == "redis://localhost:6379"
        assert restored.schema.index.name == "test-index"

    def test_roundtrip_dict_async_with_connection(self):
        idx = _make_index(cls=AsyncSearchIndex, redis_url="redis://localhost:6379")
        d = idx.to_dict(include_connection=True)
        restored = AsyncSearchIndex.from_dict(d)
        assert restored._redis_url == "redis://localhost:6379"

    def test_roundtrip_yaml(self):
        idx = _make_index(redis_url="redis://localhost:6379")
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            idx.to_yaml(path, include_connection=True)
            restored = SearchIndex.from_yaml(path)
            assert restored.schema.index.name == "test-index"
            assert restored._redis_url == "redis://localhost:6379"
        finally:
            os.unlink(path)

    def test_roundtrip_dict_with_connection_kwargs(self):
        idx = _make_index(
            redis_url="redis://localhost:6379",
            connection_kwargs={"ssl_cert_reqs": "optional"},
        )
        d = idx.to_dict(include_connection=True)
        restored = SearchIndex.from_dict(d)
        assert restored._redis_url == "redis://localhost:6379"
        assert restored._connection_kwargs == {"ssl_cert_reqs": "optional"}

    def test_roundtrip_dict_sanitized_url_skipped(self):
        """Sanitized URLs (containing ****) should not be restored — they'd
        cause auth failures."""
        idx = _make_index(redis_url="redis://:secret@localhost:6379")
        d = idx.to_dict(include_connection=True)
        assert d["_redis_url"] == "redis://:****@localhost:6379"
        restored = SearchIndex.from_dict(d)
        # The sanitized URL should be silently skipped, not restored
        assert restored._redis_url is None
