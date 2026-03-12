from pathlib import Path

import pytest

from redisvl.mcp.config import MCPConfig, load_mcp_config
from redisvl.schema import IndexSchema


def test_load_mcp_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_mcp_config("/tmp/does-not-exist.yaml")


def test_load_mcp_config_invalid_yaml(tmp_path: Path):
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text("redis_url: [", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid MCP config YAML"):
        load_mcp_config(str(config_path))


def test_load_mcp_config_env_substitution(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """
redis_url: ${REDIS_URL:-redis://localhost:6379}
index:
  name: docs
  prefix: doc
  storage_type: hash
fields:
  - name: content
    type: text
  - name: embedding
    type: vector
    attrs:
      algorithm: flat
      dims: 3
      distance_metric: cosine
      datatype: float32
vectorizer:
  class: FakeVectorizer
  model: test-model
  api_key: ${OPENAI_API_KEY}
runtime:
  text_field_name: content
  vector_field_name: embedding
  default_embed_field: content
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    config = load_mcp_config(str(config_path))

    assert config.redis_url == "redis://localhost:6379"
    assert config.vectorizer.class_name == "FakeVectorizer"
    assert config.vectorizer.model == "test-model"
    assert config.vectorizer.extra_kwargs == {"api_key": "secret"}


def test_load_mcp_config_required_env_missing(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """
redis_url: redis://localhost:6379
index:
  name: docs
  prefix: doc
  storage_type: hash
fields:
  - name: content
    type: text
  - name: embedding
    type: vector
    attrs:
      algorithm: flat
      dims: 3
      distance_metric: cosine
      datatype: float32
vectorizer:
  class: FakeVectorizer
  model: ${VECTOR_MODEL}
runtime:
  text_field_name: content
  vector_field_name: embedding
  default_embed_field: content
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("VECTOR_MODEL", raising=False)

    with pytest.raises(ValueError, match="Missing required environment variable"):
        load_mcp_config(str(config_path))


def test_mcp_config_validates_runtime_mapping():
    with pytest.raises(ValueError, match="runtime.text_field_name"):
        MCPConfig.model_validate(
            {
                "redis_url": "redis://localhost:6379",
                "index": {"name": "docs", "prefix": "doc", "storage_type": "hash"},
                "fields": [
                    {"name": "content", "type": "text"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
                "vectorizer": {"class": "FakeVectorizer", "model": "test-model"},
                "runtime": {
                    "text_field_name": "missing",
                    "vector_field_name": "embedding",
                    "default_embed_field": "content",
                },
            }
        )


def test_mcp_config_validates_vector_field_type():
    with pytest.raises(ValueError, match="runtime.vector_field_name"):
        MCPConfig.model_validate(
            {
                "redis_url": "redis://localhost:6379",
                "index": {"name": "docs", "prefix": "doc", "storage_type": "hash"},
                "fields": [
                    {"name": "content", "type": "text"},
                    {"name": "embedding", "type": "text"},
                ],
                "vectorizer": {"class": "FakeVectorizer", "model": "test-model"},
                "runtime": {
                    "text_field_name": "content",
                    "vector_field_name": "embedding",
                    "default_embed_field": "content",
                },
            }
        )


def test_mcp_config_validates_limits():
    with pytest.raises(ValueError, match="max_limit"):
        MCPConfig.model_validate(
            {
                "redis_url": "redis://localhost:6379",
                "index": {"name": "docs", "prefix": "doc", "storage_type": "hash"},
                "fields": [
                    {"name": "content", "type": "text"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
                "vectorizer": {"class": "FakeVectorizer", "model": "test-model"},
                "runtime": {
                    "text_field_name": "content",
                    "vector_field_name": "embedding",
                    "default_embed_field": "content",
                    "default_limit": 10,
                    "max_limit": 5,
                },
            }
        )


def test_mcp_config_to_index_schema():
    config = MCPConfig.model_validate(
        {
            "redis_url": "redis://localhost:6379",
            "index": {"name": "docs", "prefix": "doc", "storage_type": "hash"},
            "fields": [
                {"name": "content", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
            "vectorizer": {"class": "FakeVectorizer", "model": "test-model"},
            "runtime": {
                "text_field_name": "content",
                "vector_field_name": "embedding",
                "default_embed_field": "content",
            },
        }
    )

    schema = config.to_index_schema()

    assert isinstance(schema, IndexSchema)
    assert schema.index.name == "docs"
    assert schema.field_names == ["content", "embedding"]
