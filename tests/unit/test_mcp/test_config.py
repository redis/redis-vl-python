from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from redisvl.mcp.config import MCPConfig, load_mcp_config
from redisvl.schema import IndexSchema


def _valid_config() -> dict:
    return {
        "server": {"redis_url": "redis://localhost:6379"},
        "indexes": {
            "knowledge": {
                "redis_name": "docs-index",
                "vectorizer": {"class": "FakeVectorizer", "model": "test-model"},
                "search": {"type": "vector"},
                "runtime": {
                    "text_field_name": "content",
                    "vector_field_name": "embedding",
                    "default_embed_text_field": "content",
                },
            }
        },
    }


def _inspected_schema() -> dict:
    return {
        "index": {
            "name": "docs-index",
            "prefix": "doc",
            "storage_type": "hash",
        },
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
    }


def test_load_mcp_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_mcp_config("/tmp/does-not-exist.yaml")


def test_load_mcp_config_invalid_yaml(tmp_path: Path):
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text("server: [", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid MCP config YAML"):
        load_mcp_config(str(config_path))


def test_load_mcp_config_env_substitution(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """
server:
  redis_url: ${REDIS_URL:-redis://localhost:6379}
indexes:
    knowledge:
      redis_name: docs-index
      vectorizer:
        class: FakeVectorizer
        model: ${VECTOR_MODEL:-test-model}
        api_config:
          api_key: ${OPENAI_API_KEY}
      search:
        type: vector
      runtime:
        text_field_name: content
        vector_field_name: embedding
        default_embed_text_field: content
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    config = load_mcp_config(str(config_path))

    assert config.server.redis_url == "redis://localhost:6379"
    assert config.binding_id == "knowledge"
    assert config.redis_name == "docs-index"
    assert config.vectorizer.class_name == "FakeVectorizer"
    assert config.vectorizer.model == "test-model"
    assert config.vectorizer.extra_kwargs == {"api_config": {"api_key": "secret"}}


def test_load_mcp_config_required_env_missing(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """
server:
  redis_url: redis://localhost:6379
indexes:
    knowledge:
      redis_name: docs-index
      vectorizer:
        class: FakeVectorizer
        model: ${VECTOR_MODEL}
      search:
        type: vector
      runtime:
        text_field_name: content
        vector_field_name: embedding
        default_embed_text_field: content
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("VECTOR_MODEL", raising=False)

    with pytest.raises(ValueError, match="Missing required environment variable"):
        load_mcp_config(str(config_path))


def test_mcp_config_requires_server_redis_url():
    config = _valid_config()
    config["server"]["redis_url"] = ""

    with pytest.raises(ValueError, match="redis_url"):
        MCPConfig.model_validate(config)


@pytest.mark.parametrize(
    "indexes",
    [
        {},
        {
            "knowledge": deepcopy(_valid_config()["indexes"]["knowledge"]),
            "other": deepcopy(_valid_config()["indexes"]["knowledge"]),
        },
    ],
)
def test_mcp_config_validates_index_count(indexes):
    config = _valid_config()
    config["indexes"] = indexes

    with pytest.raises(ValueError, match="exactly one configured index binding"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_blank_binding_id():
    config = _valid_config()
    config["indexes"] = {"": deepcopy(config["indexes"]["knowledge"])}

    with pytest.raises(ValueError, match="binding id"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_blank_redis_name():
    config = _valid_config()
    config["indexes"]["knowledge"]["redis_name"] = ""

    with pytest.raises(ValueError, match="redis_name"):
        MCPConfig.model_validate(config)


def test_mcp_config_binding_helpers():
    config = MCPConfig.model_validate(_valid_config())

    assert config.binding_id == "knowledge"
    assert config.binding.redis_name == "docs-index"
    assert config.binding.search.type == "vector"
    assert config.runtime.default_embed_text_field == "content"
    assert config.vectorizer.class_name == "FakeVectorizer"
    assert config.redis_name == "docs-index"


def test_mcp_config_merges_schema_overrides_into_inspection_result():
    config_dict = _valid_config()
    config_dict["indexes"]["knowledge"]["schema_overrides"] = {
        "fields": [
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": 1536,
                    "datatype": "float32",
                    "distance_metric": "cosine",
                },
            }
        ]
    }
    inspected = _inspected_schema()
    inspected["fields"][1]["attrs"] = {"algorithm": "flat"}
    config = MCPConfig.model_validate(config_dict)

    schema = config.to_index_schema(inspected)

    assert isinstance(schema, IndexSchema)
    assert schema.index.name == "docs-index"
    assert schema.fields["embedding"].attrs.dims == 1536
    assert str(schema.fields["embedding"].attrs.algorithm).lower().endswith("flat")


def test_mcp_config_rejects_override_for_unknown_field():
    config_dict = _valid_config()
    config_dict["indexes"]["knowledge"]["schema_overrides"] = {
        "fields": [{"name": "missing", "type": "text"}]
    }
    config = MCPConfig.model_validate(config_dict)

    with pytest.raises(ValueError, match="schema_overrides.fields.*missing"):
        config.to_index_schema(_inspected_schema())


def test_mcp_config_rejects_override_type_conflict():
    config_dict = _valid_config()
    config_dict["indexes"]["knowledge"]["schema_overrides"] = {
        "fields": [{"name": "embedding", "type": "text"}]
    }
    config = MCPConfig.model_validate(config_dict)

    with pytest.raises(ValueError, match="cannot change discovered field type"):
        config.to_index_schema(_inspected_schema())


def test_mcp_config_rejects_override_path_conflict():
    config_dict = _valid_config()
    config_dict["indexes"]["knowledge"]["schema_overrides"] = {
        "fields": [{"name": "content", "type": "text", "path": "$.body"}]
    }
    inspected = {
        "index": {
            "name": "docs-index",
            "prefix": "doc",
            "storage_type": "json",
        },
        "fields": [
            {"name": "content", "type": "text", "path": "$.content"},
            {
                "name": "embedding",
                "type": "vector",
                "path": "$.embedding",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 3,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }
    config = MCPConfig.model_validate(config_dict)

    with pytest.raises(ValueError, match="cannot change discovered field path"):
        config.to_index_schema(inspected)


def test_mcp_config_validates_runtime_mapping_against_effective_schema():
    config_dict = _valid_config()
    config_dict["indexes"]["knowledge"]["runtime"]["vector_field_name"] = "content"
    config = MCPConfig.model_validate(config_dict)

    with pytest.raises(ValueError, match="runtime.vector_field_name"):
        config.to_index_schema(_inspected_schema())


def test_load_mcp_config_requires_exactly_one_binding(tmp_path: Path):
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "server": {"redis_url": "redis://localhost:6379"},
                "indexes": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one configured index binding"):
        load_mcp_config(str(config_path))


@pytest.mark.parametrize("search_type", ["vector", "fulltext", "hybrid"])
def test_mcp_config_accepts_search_types(search_type):
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": search_type}

    loaded = MCPConfig.model_validate(config)

    assert loaded.binding.search.type == search_type
    assert loaded.binding.search.params == {}


def test_mcp_config_requires_search_type():
    config = _valid_config()
    del config["indexes"]["knowledge"]["search"]["type"]

    with pytest.raises(ValueError, match="type"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_invalid_search_type():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": "semantic"}

    with pytest.raises(ValueError, match="vector|fulltext|hybrid"):
        MCPConfig.model_validate(config)


@pytest.mark.parametrize(
    ("search_type", "params"),
    [
        ("vector", {"text_scorer": "BM25STD"}),
        ("fulltext", {"normalize_vector_distance": True}),
        ("hybrid", {"normalize_vector_distance": True}),
    ],
)
def test_mcp_config_rejects_invalid_search_params(search_type, params):
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {
        "type": search_type,
        "params": params,
    }

    with pytest.raises(ValueError, match="search.params"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_linear_text_weight_without_linear_combination():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {
        "type": "hybrid",
        "params": {
            "combination_method": "RRF",
            "linear_text_weight": 0.3,
        },
    }

    with pytest.raises(ValueError, match="linear_text_weight"):
        MCPConfig.model_validate(config)


def test_mcp_config_normalizes_hybrid_linear_text_weight():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {
        "type": "hybrid",
        "params": {
            "combination_method": "LINEAR",
            "linear_text_weight": 0.3,
        },
    }

    loaded = MCPConfig.model_validate(config)

    assert loaded.binding.search.type == "hybrid"
    assert loaded.binding.search.params["linear_text_weight"] == 0.3


def test_mcp_config_allows_linear_text_weight_without_explicit_combination_method():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {
        "type": "hybrid",
        "params": {
            "linear_text_weight": 0.3,
        },
    }

    loaded = MCPConfig.model_validate(config)

    assert loaded.binding.search.type == "hybrid"
    assert loaded.binding.search.params["linear_text_weight"] == 0.3


@pytest.mark.parametrize(
    "params",
    [
        {"knn_ef_runtime": 42},
        {"vector_search_method": "RANGE", "range_radius": 0.4},
        {"combination_method": "RRF", "rrf_window": 50},
    ],
)
def test_mcp_config_rejects_native_only_hybrid_runtime_params(params):
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {
        "type": "hybrid",
        "params": params,
    }

    loaded = MCPConfig.model_validate(config)
    schema = loaded.to_index_schema(_inspected_schema())

    with pytest.raises(ValueError, match="native hybrid search support"):
        loaded.validate_search(
            schema=schema,
            supports_native_hybrid_search=False,
        )


def test_mcp_config_allows_linear_hybrid_fallback_params():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {
        "type": "hybrid",
        "params": {
            "text_scorer": "TFIDF",
            "combination_method": "LINEAR",
            "linear_text_weight": 0.3,
        },
    }

    loaded = MCPConfig.model_validate(config)
    schema = loaded.to_index_schema(_inspected_schema())

    loaded.validate_search(
        schema=schema,
        supports_native_hybrid_search=False,
    )
