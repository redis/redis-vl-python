import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from redisvl.schema.fields import BaseField
from redisvl.schema.schema import IndexInfo, IndexSchema

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


class MCPRuntimeConfig(BaseModel):
    """Runtime limits and validated field mappings for MCP requests."""

    index_mode: str = "create_if_missing"
    text_field_name: str
    vector_field_name: str
    default_embed_field: str
    default_limit: int = 10
    max_limit: int = 100
    max_upsert_records: int = 64
    skip_embedding_if_present: bool = True
    startup_timeout_seconds: int = 30
    request_timeout_seconds: int = 60
    max_concurrency: int = 16

    @model_validator(mode="after")
    def _validate_limits(self) -> "MCPRuntimeConfig":
        if self.index_mode not in {"validate_only", "create_if_missing"}:
            raise ValueError(
                "runtime.index_mode must be validate_only or create_if_missing"
            )
        if self.default_limit <= 0:
            raise ValueError("runtime.default_limit must be greater than 0")
        if self.max_limit < self.default_limit:
            raise ValueError(
                "runtime.max_limit must be greater than or equal to runtime.default_limit"
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


class MCPConfig(BaseModel):
    """Validated MCP server configuration loaded from YAML."""

    redis_url: str = Field(..., min_length=1)
    index: IndexInfo
    fields: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]
    vectorizer: MCPVectorizerConfig
    runtime: MCPRuntimeConfig

    @model_validator(mode="after")
    def _validate_runtime_mapping(self) -> "MCPConfig":
        """Ensure runtime field mappings point at explicit schema fields."""
        schema = self.to_index_schema()
        field_names = set(schema.field_names)

        if self.runtime.text_field_name not in field_names:
            raise ValueError(
                f"runtime.text_field_name '{self.runtime.text_field_name}' not found in schema"
            )

        if self.runtime.default_embed_field not in field_names:
            raise ValueError(
                f"runtime.default_embed_field '{self.runtime.default_embed_field}' not found in schema"
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

        return self

    def to_index_schema(self) -> IndexSchema:
        """Convert the MCP config schema fragment into a reusable `IndexSchema`."""
        return IndexSchema.model_validate(
            {
                "index": self.index.model_dump(mode="python"),
                "fields": self.fields,
            }
        )

    @property
    def vector_field(self) -> BaseField:
        """Return the configured vector field from the generated index schema."""
        return self.to_index_schema().fields[self.runtime.vector_field_name]

    @property
    def vector_field_dims(self) -> Optional[int]:
        """Return the configured vector dimension when the field exposes one."""
        attrs = self.vector_field.attrs
        return getattr(attrs, "dims", None)


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
        # Fail fast here so startup never proceeds with partially-resolved config.
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
