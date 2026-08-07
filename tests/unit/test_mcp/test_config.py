from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from redisvl.mcp.config import MCPConfig, builtin_tool_names, load_mcp_config
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


def _profile_dict(**overrides) -> dict:
    """Build one *unvalidated* profile dict.

    Named for its return type on purpose: this file's job is to feed raw
    payloads to the validator, so it must not be confused with a helper that
    returns an already-validated MCPCustomToolConfig.
    """
    profile = {"name": "resolved-search", "description": "Search resolved records."}
    profile.update(overrides)
    return profile


def _raw_config_with_profiles(*profiles: dict) -> dict:
    """Build an unvalidated config dict carrying the given raw profile dicts."""
    config = _valid_config()
    config["custom_tools"] = list(profiles)
    return config


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
    assert list(config.indexes) == ["knowledge"]
    binding = config.indexes["knowledge"]
    assert binding.redis_name == "docs-index"
    assert binding.vectorizer.class_name == "FakeVectorizer"
    assert binding.vectorizer.model == "test-model"
    assert binding.vectorizer.extra_kwargs == {"api_config": {"api_key": "secret"}}


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


def test_mcp_config_requires_at_least_one_binding():
    config = _valid_config()
    config["indexes"] = {}

    with pytest.raises(ValueError, match="at least one configured binding"):
        MCPConfig.model_validate(config)


def test_mcp_config_allows_multiple_bindings():
    config = _valid_config()
    config["indexes"] = {
        "knowledge": deepcopy(_valid_config()["indexes"]["knowledge"]),
        "tickets": deepcopy(_valid_config()["indexes"]["knowledge"]),
    }

    loaded = MCPConfig.model_validate(config)

    assert list(loaded.indexes) == ["knowledge", "tickets"]
    assert loaded.indexes["tickets"].redis_name == "docs-index"


def test_mcp_config_binding_defaults_for_description_and_read_only():
    config = MCPConfig.model_validate(_valid_config())

    binding = config.indexes["knowledge"]
    assert binding.description is None
    assert binding.read_only is False


def test_mcp_config_binding_accepts_description_and_read_only():
    config = _valid_config()
    config["indexes"]["knowledge"]["description"] = "Product docs and runbooks"
    config["indexes"]["knowledge"]["read_only"] = True

    binding = MCPConfig.model_validate(config).indexes["knowledge"]

    assert binding.description == "Product docs and runbooks"
    assert binding.read_only is True


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


def test_mcp_config_binding_exposes_index_settings():
    config = MCPConfig.model_validate(_valid_config())

    binding = config.indexes["knowledge"]
    assert binding.redis_name == "docs-index"
    assert binding.search.type == "vector"
    assert binding.runtime.default_embed_text_field == "content"
    assert binding.vectorizer.class_name == "FakeVectorizer"


def test_vector_search_config_can_omit_text_field_name():
    config = _valid_config()
    del config["indexes"]["knowledge"]["runtime"]["text_field_name"]

    binding = MCPConfig.model_validate(config).indexes["knowledge"]

    assert binding.search.type == "vector"
    assert binding.runtime.text_field_name is None


def test_fulltext_config_can_omit_vector_settings_and_vectorizer():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": "fulltext"}
    del config["indexes"]["knowledge"]["vectorizer"]
    del config["indexes"]["knowledge"]["runtime"]["vector_field_name"]
    del config["indexes"]["knowledge"]["runtime"]["default_embed_text_field"]

    binding = MCPConfig.model_validate(config).indexes["knowledge"]

    assert binding.search.type == "fulltext"
    assert binding.vectorizer is None
    assert binding.runtime.vector_field_name is None
    assert binding.runtime.default_embed_text_field is None


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

    schema = config.indexes["knowledge"].to_index_schema(inspected)

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
        config.indexes["knowledge"].to_index_schema(_inspected_schema())


def test_mcp_config_rejects_override_type_conflict():
    config_dict = _valid_config()
    config_dict["indexes"]["knowledge"]["schema_overrides"] = {
        "fields": [{"name": "embedding", "type": "text"}]
    }
    config = MCPConfig.model_validate(config_dict)

    with pytest.raises(ValueError, match="cannot change discovered field type"):
        config.indexes["knowledge"].to_index_schema(_inspected_schema())


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
        config.indexes["knowledge"].to_index_schema(inspected)


def test_mcp_config_validates_runtime_mapping_against_effective_schema():
    config_dict = _valid_config()
    config_dict["indexes"]["knowledge"]["runtime"]["vector_field_name"] = "content"
    config = MCPConfig.model_validate(config_dict)

    with pytest.raises(ValueError, match="runtime.vector_field_name"):
        config.indexes["knowledge"].to_index_schema(_inspected_schema())


def test_fulltext_config_does_not_require_vector_mapping_in_schema():
    config_dict = _valid_config()
    config_dict["indexes"]["knowledge"]["search"] = {"type": "fulltext"}
    del config_dict["indexes"]["knowledge"]["vectorizer"]
    del config_dict["indexes"]["knowledge"]["runtime"]["vector_field_name"]
    del config_dict["indexes"]["knowledge"]["runtime"]["default_embed_text_field"]
    config = MCPConfig.model_validate(config_dict)

    schema = config.indexes["knowledge"].to_index_schema(_inspected_schema())

    assert isinstance(schema, IndexSchema)


def test_load_mcp_config_requires_at_least_one_binding(tmp_path: Path):
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

    with pytest.raises(ValueError, match="at least one configured binding"):
        load_mcp_config(str(config_path))


@pytest.mark.parametrize("search_type", ["vector", "fulltext", "hybrid"])
def test_mcp_config_accepts_search_types(search_type):
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": search_type}

    loaded = MCPConfig.model_validate(config)

    assert loaded.indexes["knowledge"].search.type == search_type
    assert loaded.indexes["knowledge"].search.params == {}


def test_mcp_config_requires_search_type():
    config = _valid_config()
    del config["indexes"]["knowledge"]["search"]["type"]

    with pytest.raises(ValueError, match="type"):
        MCPConfig.model_validate(config)


def test_fulltext_config_requires_text_field_name():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": "fulltext"}
    del config["indexes"]["knowledge"]["runtime"]["text_field_name"]
    del config["indexes"]["knowledge"]["vectorizer"]
    del config["indexes"]["knowledge"]["runtime"]["vector_field_name"]
    del config["indexes"]["knowledge"]["runtime"]["default_embed_text_field"]

    with pytest.raises(ValueError, match="runtime.text_field_name"):
        MCPConfig.model_validate(config)


def test_hybrid_config_requires_vector_field_and_vectorizer():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": "hybrid"}
    del config["indexes"]["knowledge"]["runtime"]["vector_field_name"]

    with pytest.raises(ValueError, match="runtime.vector_field_name"):
        MCPConfig.model_validate(config)

    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": "hybrid"}
    del config["indexes"]["knowledge"]["vectorizer"]

    with pytest.raises(ValueError, match="vectorizer"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_invalid_search_type():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": "semantic"}

    with pytest.raises(ValueError, match="vector|fulltext|hybrid"):
        MCPConfig.model_validate(config)


def test_server_side_embedding_requires_vector_field_and_vectorizer():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": "fulltext"}
    del config["indexes"]["knowledge"]["runtime"]["vector_field_name"]

    with pytest.raises(
        ValueError, match="default_embed_text_field requires runtime.vector_field_name"
    ):
        MCPConfig.model_validate(config)

    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": "fulltext"}
    del config["indexes"]["knowledge"]["vectorizer"]

    with pytest.raises(
        ValueError, match="default_embed_text_field requires vectorizer"
    ):
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

    assert loaded.indexes["knowledge"].search.type == "hybrid"
    assert loaded.indexes["knowledge"].search.params["linear_text_weight"] == 0.3


def test_mcp_config_allows_linear_text_weight_without_explicit_combination_method():
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {
        "type": "hybrid",
        "params": {
            "linear_text_weight": 0.3,
        },
    }

    loaded = MCPConfig.model_validate(config)

    assert loaded.indexes["knowledge"].search.type == "hybrid"
    assert loaded.indexes["knowledge"].search.params["linear_text_weight"] == 0.3


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
    schema = loaded.indexes["knowledge"].to_index_schema(_inspected_schema())

    with pytest.raises(ValueError, match="native hybrid search support"):
        loaded.indexes["knowledge"].validate_search(
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
    schema = loaded.indexes["knowledge"].to_index_schema(_inspected_schema())

    loaded.indexes["knowledge"].validate_search(
        schema=schema,
        supports_native_hybrid_search=False,
    )


@pytest.mark.parametrize("search_type", ["fulltext", "hybrid"])
def test_mcp_config_rejects_a_text_field_that_is_actually_the_vector_field(search_type):
    """Membership in the schema is not enough -- the field must be text-searchable.

    Pointing `text_field_name` at the vector field passes a name-only check while
    leaving the default projection (every *non-vector* field) empty. An empty
    projection reaches Redis as no RETURN clause, which returns every stored
    field including the embedding, so this has to fail at startup.
    """
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": search_type}
    config["indexes"]["knowledge"]["runtime"]["text_field_name"] = "embedding"

    binding = MCPConfig.model_validate(config).indexes["knowledge"]

    # `to_index_schema` validates the runtime mapping against the effective
    # schema, so startup fails here rather than at the first request.
    with pytest.raises(ValueError, match="is a vector field"):
        binding.to_index_schema(_inspected_schema())


def test_mcp_config_still_accepts_a_real_text_field():
    """The control: a genuine text field must keep validating."""
    config = _valid_config()
    config["indexes"]["knowledge"]["search"] = {"type": "fulltext"}
    config["indexes"]["knowledge"]["runtime"]["text_field_name"] = "content"

    binding = MCPConfig.model_validate(config).indexes["knowledge"]
    schema = binding.to_index_schema(_inspected_schema())

    binding.validate_runtime_mapping(schema)


def test_mcp_config_builtin_tools_default_to_enabled():
    config = MCPConfig.model_validate(_valid_config())

    assert config.server.builtin_tools == {}
    for tool_name in builtin_tool_names():
        assert config.server.builtin_tool_enabled(tool_name) is True


def test_mcp_config_builtin_tools_can_disable_a_builtin():
    config = _valid_config()
    config["server"]["builtin_tools"] = {
        "search-records": "disabled",
        "list-indexes": "enabled",
    }

    loaded = MCPConfig.model_validate(config)

    assert loaded.server.builtin_tool_enabled("search-records") is False
    assert loaded.server.builtin_tool_enabled("list-indexes") is True
    # Unmentioned built-ins stay enabled.
    assert loaded.server.builtin_tool_enabled("upsert-records") is True


def test_mcp_config_rejects_unknown_builtin_tool_names():
    config = _valid_config()
    # Underscores instead of hyphens -- the most likely typo, and one that would
    # otherwise disable nothing while reading as though it had.
    config["server"]["builtin_tools"] = {"search_records": "disabled"}

    with pytest.raises(
        ValueError, match="server.builtin_tools contains unknown tool names"
    ):
        MCPConfig.model_validate(config)


def test_load_mcp_config_parses_builtin_tools_from_yaml(tmp_path: Path):
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """
server:
  redis_url: redis://localhost:6379
  builtin_tools:
    upsert-records: disabled
indexes:
    knowledge:
      redis_name: docs-index
      search:
        type: fulltext
      runtime:
        text_field_name: content
""".strip(),
        encoding="utf-8",
    )

    config = load_mcp_config(str(config_path))

    assert config.server.builtin_tool_enabled("upsert-records") is False
    assert config.server.builtin_tool_enabled("search-records") is True


def test_mcp_config_defaults_to_no_custom_tools():
    config = MCPConfig.model_validate(_valid_config())

    assert config.custom_tools == []


def test_mcp_config_custom_tool_defaults():
    config = MCPConfig.model_validate(_raw_config_with_profiles(_profile_dict()))

    profile = config.custom_tools[0]
    assert profile.name == "resolved-search"
    assert profile.kind == "profile"
    assert profile.based_on == "search-records"
    # An unpinned profile is legal with one binding and resolves to it.
    assert profile.index is None
    assert profile.suppress_schema_hints is False
    assert profile.lock.return_fields is None
    assert profile.lock.filter is None
    assert profile.params == {}
    assert config.resolved_profile_index(profile) == "knowledge"


@pytest.mark.parametrize(
    "name",
    [
        # One violation per position the pattern constrains: the anchored first
        # character, and the body character class.
        "1foo",
        "shop.foo",
    ],
)
def test_mcp_config_rejects_invalid_custom_tool_names(name):
    with pytest.raises(ValueError, match="is invalid"):
        MCPConfig.model_validate(_raw_config_with_profiles(_profile_dict(name=name)))


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        # The pattern is `^[a-z][a-z0-9_-]{0,63}$`, so 64 characters is the
        # inclusive ceiling and 65 is the first rejection. Nothing else pins this
        # bound, and an off-by-one in the quantifier is otherwise invisible.
        ("a" + "b" * 63, True),
        ("a" + "b" * 64, False),
    ],
)
def test_mcp_config_bounds_custom_tool_name_length_at_64_characters(name, valid):
    config = _raw_config_with_profiles(_profile_dict(name=name))

    if valid:
        assert MCPConfig.model_validate(config).custom_tools[0].name == name
    else:
        with pytest.raises(ValueError, match="is invalid"):
            MCPConfig.model_validate(config)


def test_mcp_config_rejects_custom_tool_names_that_collide_with_builtins():
    # One built-in stands in for the rest: the check is a membership test against
    # builtin_tool_names(), not per-name logic.
    name = "search-records"
    assert name in builtin_tool_names()

    with pytest.raises(ValueError, match="collides with a built-in tool"):
        MCPConfig.model_validate(_raw_config_with_profiles(_profile_dict(name=name)))


def test_mcp_config_rejects_unknown_custom_tool_params():
    config = _raw_config_with_profiles(
        _profile_dict(params={"search_type": {"expose": False}})
    )

    with pytest.raises(ValueError, match="params contains unknown arguments"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_max_on_params_other_than_limit():
    # `limit` is the sole param with a cap; every other name takes the same
    # rejection branch, so one stands in for all of them.
    config = _raw_config_with_profiles(_profile_dict(params={"offset": {"max": 5}}))

    with pytest.raises(ValueError, match="does not support 'max'"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_non_positive_param_max():
    # 0 is the boundary: it is the largest value the `> 0` check must still refuse.
    config = _raw_config_with_profiles(_profile_dict(params={"limit": {"max": 0}}))

    with pytest.raises(ValueError, match="max must be greater than 0"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_hiding_the_query_param():
    config = _raw_config_with_profiles(
        _profile_dict(params={"query": {"expose": False}})
    )

    with pytest.raises(ValueError, match="params.query cannot be hidden"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_locking_return_fields_while_exposing_them():
    config = _raw_config_with_profiles(
        _profile_dict(
            lock={"return_fields": ["content"]},
            params={"return_fields": {"expose": True}},
        )
    )

    with pytest.raises(ValueError, match="cannot both lock return_fields"):
        MCPConfig.model_validate(config)


def test_mcp_config_allows_locking_return_fields_when_they_are_explicitly_hidden():
    config = _raw_config_with_profiles(
        _profile_dict(
            lock={"return_fields": ["content"]},
            params={"return_fields": {"expose": False}},
        )
    )

    profile = MCPConfig.model_validate(config).custom_tools[0]

    assert profile.lock.return_fields == ["content"]
    assert profile.param_exposed("return_fields") is False


def test_mcp_config_allows_locking_a_filter_while_the_caller_filter_stays_exposed():
    config = _raw_config_with_profiles(
        _profile_dict(
            lock={"filter": {"field": "content", "op": "like", "value": "jam*"}},
            params={"filter": {"expose": True}},
        )
    )

    profile = MCPConfig.model_validate(config).custom_tools[0]

    # Unlike return_fields, locked-plus-exposed is the intended narrowing case.
    assert profile.param_exposed("filter") is True


def test_mcp_config_rejects_empty_locked_return_fields():
    config = _raw_config_with_profiles(_profile_dict(lock={"return_fields": []}))

    with pytest.raises(ValueError, match="must contain at least one field"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_blank_locked_return_field_names():
    # Whitespace rather than "": the empty string is already falsy, so only this
    # one requires the check to `.strip()` before testing emptiness.
    config = _raw_config_with_profiles(_profile_dict(lock={"return_fields": ["   "]}))

    with pytest.raises(ValueError, match="must contain non-empty strings"):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_duplicate_custom_tool_names():
    config = _raw_config_with_profiles(
        _profile_dict(), _profile_dict(description="Another description.")
    )

    with pytest.raises(ValueError, match="contains duplicate tool name"):
        MCPConfig.model_validate(config)


def test_mcp_config_requires_custom_tool_index_when_multiple_bindings_exist():
    config = _raw_config_with_profiles(_profile_dict())
    config["indexes"]["tickets"] = deepcopy(config["indexes"]["knowledge"])

    with pytest.raises(
        ValueError, match="must set 'index' when multiple indexes are configured"
    ):
        MCPConfig.model_validate(config)


def test_mcp_config_rejects_custom_tool_index_naming_an_unknown_binding():
    config = _raw_config_with_profiles(_profile_dict(index="missing"))

    with pytest.raises(ValueError, match="references unknown index"):
        MCPConfig.model_validate(config)


def test_mcp_config_resolves_a_pinned_custom_tool_index():
    config = _raw_config_with_profiles(_profile_dict(index="tickets"))
    config["indexes"]["tickets"] = deepcopy(config["indexes"]["knowledge"])

    loaded = MCPConfig.model_validate(config)

    assert loaded.resolved_profile_index(loaded.custom_tools[0]) == "tickets"


def test_mcp_config_param_exposed_hides_return_fields_implicitly_when_locked():
    config = _raw_config_with_profiles(
        _profile_dict(lock={"return_fields": ["content"]})
    )

    profile = MCPConfig.model_validate(config).custom_tools[0]

    # Locking implies hiding without the author having to say so twice.
    assert profile.param_exposed("return_fields") is False
    assert profile.param_exposed("filter") is True


@pytest.mark.parametrize(
    ("label", "profile_patch"),
    [
        # One case per model that declares `extra="forbid"` -- the profile itself,
        # its `lock`, and a per-param policy. There are exactly three, and within
        # any one of them every misspelling takes the identical pydantic branch.
        ("profile", {"basedon": "search-records"}),
        ("lock", {"lock": {"return_field": ["content"]}}),
        ("params policy", {"params": {"limit": {"exposed": True}}}),
    ],
)
def test_custom_tools_rejects_misspelled_keys(label, profile_patch):
    del label
    config = _valid_config()
    profile = {"name": "resolved-search", "description": "Search resolved records."}
    profile.update(profile_patch)
    config["custom_tools"] = [profile]

    # A dropped key would leave a tool that reads as locked in config while
    # enforcing nothing, so unrecognized keys must fail rather than be ignored.
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        MCPConfig.model_validate(config)


def test_custom_tools_rejects_a_limit_cap_above_the_bindings_max_limit():
    config = _valid_config()
    config["indexes"]["knowledge"]["runtime"]["default_limit"] = 5
    config["indexes"]["knowledge"]["runtime"]["max_limit"] = 5
    config["custom_tools"] = [
        {
            "name": "capped-search",
            "description": "Search records.",
            "params": {"limit": {"expose": False, "max": 10}},
        }
    ]

    # With `limit` hidden the cap becomes the request size, so a cap the binding
    # can never satisfy would make every call fail. Catch it before startup.
    with pytest.raises(ValueError, match="exceeds the bound index's"):
        MCPConfig.model_validate(config)


@pytest.mark.parametrize("name", ["redisvl-search", "redisvl_search"])
def test_custom_tools_rejects_both_reserved_prefix_separators(name):
    config = _valid_config()
    config["custom_tools"] = [{"name": name, "description": "Search records."}]

    # The name pattern permits either separator, so both spellings of the
    # reserved prefix have to be refused.
    with pytest.raises(ValueError, match="reserved prefix"):
        MCPConfig.model_validate(config)


def test_load_mcp_config_parses_custom_tools_from_yaml(tmp_path: Path):
    """The whole point is YAML authoring, so cover the real load path once."""
    config_path = tmp_path / "mcp.yaml"
    config_path.write_text(
        """
server:
  redis_url: redis://localhost:6379
  builtin_tools:
    search-records: disabled
indexes:
    knowledge:
      redis_name: docs-index
      search:
        type: fulltext
      runtime:
        text_field_name: content
custom_tools:
  - name: resolved-search
    description: Search resolved records.
    lock:
      return_fields: [content]
      filter:
        field: category
        op: eq
        value: resolved
    params:
      limit:
        max: 5
""".strip(),
        encoding="utf-8",
    )

    config = load_mcp_config(str(config_path))

    profile = config.custom_tools[0]
    assert profile.name == "resolved-search"
    assert profile.lock.return_fields == ["content"]
    assert profile.lock.filter == {
        "field": "category",
        "op": "eq",
        "value": "resolved",
    }
    assert profile.param_max("limit") == 5
    # Disabling the built-in a profile supersedes is the motivating pairing, so
    # confirm both halves survive one load.
    assert config.server.builtin_tool_enabled("search-records") is False
