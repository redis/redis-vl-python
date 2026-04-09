import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from redisvl.schema.fields import BaseField
from redisvl.schema.schema import IndexSchema

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")
_RESERVED_SCORE_METADATA_FIELDS = frozenset(
    {"vector_distance", "__score", "text_score", "vector_similarity", "hybrid_score"}
)


def reserved_score_metadata_field_names() -> frozenset[str]:
    """Return MCP-reserved score metadata field names."""
    return _RESERVED_SCORE_METADATA_FIELDS


class MCPRuntimeConfig(BaseModel):
    """Runtime limits and validated field mappings for MCP requests."""

    text_field_name: str = Field(..., min_length=1)
    vector_field_name: str = Field(..., min_length=1)
    default_embed_text_field: str = Field(..., min_length=1)
    default_limit: int = 10
    max_limit: int = 100
    max_result_window: int = 1000
    max_upsert_records: int = 64
    skip_embedding_if_present: bool = True
    startup_timeout_seconds: int = 30
    request_timeout_seconds: int = 60
    max_concurrency: int = 16

    @model_validator(mode="after")
    def _validate_limits(self) -> "MCPRuntimeConfig":
        """Validate runtime bounds during config load."""
        if self.default_limit <= 0:
            raise ValueError("runtime.default_limit must be greater than 0")
        if self.max_limit < self.default_limit:
            raise ValueError(
                "runtime.max_limit must be greater than or equal to runtime.default_limit"
            )
        if self.max_result_window <= 0:
            raise ValueError("runtime.max_result_window must be greater than 0")
        if self.max_result_window < self.max_limit:
            raise ValueError(
                "runtime.max_result_window must be greater than or equal to runtime.max_limit"
            )
        if self.max_upsert_records <= 0:
            raise ValueError("runtime.max_upsert_records must be greater than 0")
        if self.startup_timeout_seconds <= 0:
            raise ValueError("runtime.startup_timeout_seconds must be greater than 0")
        if self.request_timeout_seconds <= 0:
            raise ValueError("runtime.request_timeout_seconds must be greater than 0")
        if self.max_concurrency <= 0:
            raise ValueError("runtime.max_concurrency must be greater than 0")
        return self


class MCPVectorizerConfig(BaseModel):
    """Vectorizer constructor contract loaded from YAML."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    class_name: str = Field(alias="class", min_length=1)
    model: str = Field(..., min_length=1)

    @property
    def extra_kwargs(self) -> Dict[str, Any]:
        """Return vectorizer kwargs other than the normalized `class` and `model`."""
        return dict(self.model_extra or {})

    def to_init_kwargs(self) -> Dict[str, Any]:
        """Build kwargs suitable for directly instantiating the vectorizer."""
        return {"model": self.model, **self.extra_kwargs}


class MCPServerConfig(BaseModel):
    """Server-level bootstrap configuration."""

    redis_url: str = Field(..., min_length=1)


class MCPIndexSearchConfig(BaseModel):
    """Configured search mode and query tuning for the bound index.

    The MCP request contract only exposes query text, filtering, pagination, and
    field projection. Search mode and query-tuning behavior are owned entirely by
    YAML config and validated here.
    """

    type: Literal["vector", "fulltext", "hybrid"]
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> "MCPIndexSearchConfig":
        """Reject params that do not belong to the configured search mode."""
        allowed_params = {
            "vector": {
                "hybrid_policy",
                "batch_size",
                "ef_runtime",
                "epsilon",
                "search_window_size",
                "use_search_history",
                "search_buffer_capacity",
                "normalize_vector_distance",
            },
            "fulltext": {
                "text_scorer",
                "stopwords",
                "text_weights",
            },
            "hybrid": {
                "text_scorer",
                "stopwords",
                "text_weights",
                "vector_search_method",
                "knn_ef_runtime",
                "range_radius",
                "range_epsilon",
                "combination_method",
                "rrf_window",
                "rrf_constant",
                "linear_text_weight",
            },
        }
        invalid_keys = sorted(set(self.params) - allowed_params[self.type])
        if invalid_keys:
            raise ValueError(
                "search.params contains keys incompatible with "
                f"search.type '{self.type}': {', '.join(invalid_keys)}"
            )

        if "linear_text_weight" in self.params and self.params.get(
            "combination_method"
        ) not in (None, "LINEAR"):
            raise ValueError(
                "search.params.linear_text_weight requires combination_method to be LINEAR"
            )
        return self

    def to_query_params(self) -> Dict[str, Any]:
        """Return normalized query kwargs exactly as configured."""
        return dict(self.params)

    def validate_runtime_capabilities(
        self, *, supports_native_hybrid_search: bool
    ) -> None:
        """Fail startup when hybrid config depends on native-only FT.SEARCH params."""
        if self.type != "hybrid" or supports_native_hybrid_search:
            return

        unsupported_params = set()
        if self.params.get("combination_method") not in (None, "LINEAR"):
            unsupported_params.add("combination_method")
        unsupported_params.update(
            key
            for key in (
                "vector_search_method",
                "knn_ef_runtime",
                "range_radius",
                "range_epsilon",
                "rrf_window",
                "rrf_constant",
            )
            if key in self.params
        )

        if unsupported_params:
            unsupported_list = ", ".join(sorted(unsupported_params))
            raise ValueError(
                "search.params requires native hybrid search support for: "
                f"{unsupported_list}"
            )


class MCPSchemaOverrideField(BaseModel):
    """Allowed schema override fragment for one already-discovered field."""

    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    path: Optional[str] = None
    attrs: Dict[str, Any] = Field(default_factory=dict)


class MCPSchemaOverrides(BaseModel):
    """Optional field-level schema patches used to fill inspection gaps."""

    fields: list[MCPSchemaOverrideField] = Field(default_factory=list)


class MCPIndexBindingConfig(BaseModel):
    """The sole configured v1 index binding."""

    redis_name: str = Field(..., min_length=1)
    vectorizer: MCPVectorizerConfig
    search: MCPIndexSearchConfig
    runtime: MCPRuntimeConfig
    schema_overrides: MCPSchemaOverrides = Field(default_factory=MCPSchemaOverrides)


class MCPConfig(BaseModel):
    """Validated MCP server configuration loaded from YAML."""

    server: MCPServerConfig
    indexes: Dict[str, MCPIndexBindingConfig]

    @model_validator(mode="after")
    def _validate_bindings(self) -> "MCPConfig":
        """Validate that there is exactly one configured logical binding."""
        if len(self.indexes) != 1:
            raise ValueError(
                "indexes must contain exactly one configured index binding"
            )

        binding_id = next(iter(self.indexes))
        if not binding_id.strip():
            raise ValueError("indexes binding id must be non-blank")
        return self

    @property
    def binding_id(self) -> str:
        """Return the single logical binding identifier configured for v1."""
        return next(iter(self.indexes))

    @property
    def binding(self) -> MCPIndexBindingConfig:
        """Return the sole configured binding."""
        return self.indexes[self.binding_id]

    @property
    def runtime(self) -> MCPRuntimeConfig:
        """Expose the sole binding's runtime config for phase 1."""
        return self.binding.runtime

    @property
    def vectorizer(self) -> MCPVectorizerConfig:
        """Expose the sole binding's vectorizer config for phase 1."""
        return self.binding.vectorizer

    @property
    def search(self) -> MCPIndexSearchConfig:
        """Expose the sole binding's configured search behavior."""
        return self.binding.search

    @property
    def redis_name(self) -> str:
        """Return the existing Redis index name that must be inspected at startup."""
        return self.binding.redis_name

    def inspected_schema_from_index_info(
        self, index_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a schema dict from FT.INFO while preserving discovered field identity.

        RedisVL's generic FT.INFO conversion omits vector fields when their attrs are
        incomplete on older Redis versions. MCP needs those field identities to survive
        so schema overrides can patch the missing attrs during startup.
        """
        from redisvl.redis.connection import convert_index_info_to_schema

        schema_dict = convert_index_info_to_schema(index_info)
        discovered_fields = {
            field["name"]: field
            for field in schema_dict.get("fields", [])
            if isinstance(field, dict) and "name" in field
        }

        storage_type = index_info["index_definition"][1].lower()
        for raw_field in index_info.get("attributes", []):
            name = raw_field[1] if storage_type == "hash" else raw_field[3]
            if name in discovered_fields:
                continue

            field = {
                "name": name,
                "type": str(raw_field[5]).lower(),
            }
            if storage_type == "json":
                field["path"] = raw_field[1]

            # Keep discovered field identity even when FT.INFO omitted attrs.
            schema_dict.setdefault("fields", []).append(field)

        return schema_dict

    def merge_schema_overrides(
        self, inspected_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply validated schema overrides without allowing identity changes."""
        merged_schema = deepcopy(inspected_schema)
        merged_fields = merged_schema.setdefault("fields", [])
        discovered_fields = {
            field["name"]: field
            for field in merged_fields
            if isinstance(field, dict) and "name" in field
        }

        for override in self.binding.schema_overrides.fields:
            discovered = discovered_fields.get(override.name)
            if discovered is None:
                raise ValueError(
                    f"schema_overrides.fields '{override.name}' not found in inspected schema"
                )

            discovered_type = str(discovered.get("type", "")).lower()
            override_type = override.type.lower()
            if discovered_type != override_type:
                raise ValueError(
                    f"schema_overrides.fields '{override.name}' cannot change discovered field type"
                )

            discovered_path = discovered.get("path")
            if override.path is not None and override.path != discovered_path:
                raise ValueError(
                    f"schema_overrides.fields '{override.name}' cannot change discovered field path"
                )

            if override.attrs:
                merged_attrs = dict(discovered.get("attrs", {}))
                merged_attrs.update(override.attrs)
                discovered["attrs"] = merged_attrs

        return merged_schema

    def validate_runtime_mapping(self, schema: IndexSchema) -> None:
        """Ensure runtime mappings point at explicit fields in the effective schema."""
        field_names = set(schema.field_names)

        if self.runtime.text_field_name not in field_names:
            raise ValueError(
                f"runtime.text_field_name '{self.runtime.text_field_name}' not found in schema"
            )

        if self.runtime.default_embed_text_field not in field_names:
            raise ValueError(
                "runtime.default_embed_text_field "
                f"'{self.runtime.default_embed_text_field}' not found in schema"
            )

        vector_field = schema.fields.get(self.runtime.vector_field_name)
        if vector_field is None:
            raise ValueError(
                f"runtime.vector_field_name '{self.runtime.vector_field_name}' not found in schema"
            )
        if vector_field.type != "vector":
            raise ValueError(
                f"runtime.vector_field_name '{self.runtime.vector_field_name}' must reference a vector field"
            )

    def to_index_schema(self, inspected_schema: Dict[str, Any]) -> IndexSchema:
        """Apply overrides to an inspected schema and validate the effective result."""
        merged_schema = self.merge_schema_overrides(inspected_schema)
        schema = IndexSchema.model_validate(merged_schema)
        self.validate_runtime_mapping(schema)
        return schema

    def get_vector_field(self, schema: IndexSchema) -> BaseField:
        """Return the effective vector field from a validated schema."""
        return schema.fields[self.runtime.vector_field_name]

    def get_vector_field_dims(self, schema: IndexSchema) -> Optional[int]:
        """Return the effective vector dimensions when the field exposes them."""
        attrs = self.get_vector_field(schema).attrs
        return getattr(attrs, "dims", None)

    def validate_search(
        self,
        *,
        schema: IndexSchema,
        supports_native_hybrid_search: bool,
    ) -> None:
        """Validate configured search behavior against current runtime support."""
        self.search.validate_runtime_capabilities(
            supports_native_hybrid_search=supports_native_hybrid_search
        )
        conflicting_fields = sorted(
            set(schema.field_names) & reserved_score_metadata_field_names()
        )
        if conflicting_fields:
            raise ValueError(
                "Schema fields conflict with MCP-reserved score metadata names "
                f"for search.type '{self.search.type}': {', '.join(conflicting_fields)}"
            )


def _substitute_env(value: Any) -> Any:
    """Recursively resolve `${VAR}` and `${VAR:-default}` placeholders."""
    if isinstance(value, dict):
        return {key: _substitute_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        raise ValueError(f"Missing required environment variable: {name}")

    return _ENV_PATTERN.sub(replace, value)


def load_mcp_config(path: str) -> MCPConfig:
    """Load, substitute, and validate the MCP YAML configuration file."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"MCP config file {path} does not exist")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid MCP config YAML: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ValueError("Invalid MCP config YAML: root document must be a mapping")

    substituted = _substitute_env(raw_data)
    return MCPConfig.model_validate(substituted)
