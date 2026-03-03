# RedisVL MCP Server Specification

## Overview

This specification defines the implementation of a Model Context Protocol (MCP) server for RedisVL. The MCP server enables AI agents and LLM applications to interact with Redis as a vector database through a standardized protocol.

### Goals

1. Expose RedisVL's vector search capabilities to MCP-compatible clients (Claude Desktop, Claude Agents SDK, etc.)
2. Provide tools for semantic search, full-text search, hybrid search, and data upsert
3. Integrate seamlessly with the existing RedisVL architecture
4. Support the `uvx --from redisvl rvl mcp` pattern for easy deployment

### References

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [FastMCP Library](https://github.com/jlowin/fastmcp)
- [Qdrant MCP Server](https://github.com/qdrant/mcp-server-qdrant) - Similar scope reference
- [Redis Agent Memory Server](https://github.com/redis-developer/agent-memory-server) - Implementation patterns

---

## Architecture

### Module Structure

```
redisvl/
├── mcp/
│   ├── __init__.py           # Public exports
│   ├── server.py             # RedisVLMCPServer class (extends FastMCP)
│   ├── settings.py           # MCPSettings (pydantic-settings)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search.py         # Search tool implementation
│   │   └── upsert.py         # Upsert tool implementation
│   └── utils.py              # Helper functions
├── cli/
│   └── ... (existing)        # Add `mcp` subcommand
```

### Dependencies

The MCP functionality is an **optional dependency group**:

```toml
# pyproject.toml
[project.optional-dependencies]
mcp = [
    "mcp>=1.9.0",              # MCP SDK with FastMCP
]
```

Installation: `pip install redisvl[mcp]`

### Core Components

1. **RedisVLMCPServer**: Main server class extending `FastMCP`
2. **MCPSettings**: Configuration via environment variables (pydantic-settings)
3. **Tool implementations**: Search and upsert operations
4. **CLI integration**: `rvl mcp` subcommand

---

## Configuration (MCPSettings)

Settings are configured via environment variables, following the pattern established by Qdrant MCP and Agent Memory Server.

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDISVL_MCP_CONFIG` | str | (required) | Path to MCP configuration YAML file |
| `REDISVL_MCP_READ_ONLY` | bool | `false` | Disable upsert tool when true |
| `REDISVL_MCP_TOOL_SEARCH_DESCRIPTION` | str | (see below) | Custom search tool description |
| `REDISVL_MCP_TOOL_UPSERT_DESCRIPTION` | str | (see below) | Custom upsert tool description |

### MCP Configuration File

All MCP server configuration is consolidated into a **single YAML file** that includes Redis connection, index schema, and vectorizer settings. This simplifies deployment and keeps related configuration together.

#### Configuration File Format

```yaml
# mcp_config.yaml

# Redis connection
redis_url: redis://localhost:6379

# Index schema (inline, same format as existing RedisVL schemas)
index:
  name: my_index
  prefix: doc
  storage_type: hash    # or "json"

fields:
  - name: content
    type: text
  - name: category
    type: tag
  - name: embedding
    type: vector
    attrs:
      algorithm: hnsw
      dims: 1536
      distance_metric: cosine
      datatype: float32

# Vectorizer configuration
# Use the exact class name from redisvl.utils.vectorize
vectorizer:
  class: OpenAITextVectorizer      # Required: vectorizer class name
  model: text-embedding-3-small    # Required: model name

  # Additional kwargs passed directly to the vectorizer constructor
  # Most providers use environment variables by default for API keys
```

#### Provider-Specific Vectorizer Examples

```yaml
# OpenAI (simplest - uses OPENAI_API_KEY env var automatically)
vectorizer:
  class: OpenAITextVectorizer
  model: text-embedding-3-small

# Azure OpenAI
vectorizer:
  class: AzureOpenAITextVectorizer
  model: text-embedding-ada-002
  api_key: ${AZURE_OPENAI_API_KEY}
  api_version: "2024-02-01"
  azure_endpoint: ${AZURE_OPENAI_ENDPOINT}

# AWS Bedrock
vectorizer:
  class: BedrockTextVectorizer
  model: amazon.titan-embed-text-v1
  region_name: us-east-1
  # Uses AWS credentials from environment/IAM role by default

# Google VertexAI
vectorizer:
  class: VertexAITextVectorizer
  model: textembedding-gecko@003
  project_id: ${GCP_PROJECT_ID}
  location: us-central1

# HuggingFace (local embeddings)
vectorizer:
  class: HFTextVectorizer
  model: sentence-transformers/all-MiniLM-L6-v2

# Cohere
vectorizer:
  class: CohereTextVectorizer
  model: embed-english-v3.0
  # Uses COHERE_API_KEY env var automatically

# Mistral
vectorizer:
  class: MistralAITextVectorizer
  model: mistral-embed
  # Uses MISTRAL_API_KEY env var automatically

# VoyageAI
vectorizer:
  class: VoyageAITextVectorizer
  model: voyage-2
  # Uses VOYAGE_API_KEY env var automatically
```

### Configuration Loader

```python
# redisvl/mcp/config.py
from typing import Any, Dict, Optional
import os
import re
import yaml

from redisvl.schema import IndexSchema

def load_mcp_config(config_path: str) -> Dict[str, Any]:
    """Load MCP config with environment variable substitution."""
    with open(config_path) as f:
        content = f.read()

    # Substitute ${VAR} patterns with environment variables
    def replace_env(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    content = re.sub(r'\$\{(\w+)\}', replace_env, content)
    return yaml.safe_load(content)

def create_index_schema(config: Dict[str, Any]) -> IndexSchema:
    """Create IndexSchema from the index/fields portion of config."""
    schema_dict = {
        "index": config["index"],
        "fields": config["fields"],
    }
    return IndexSchema.from_dict(schema_dict)

def create_vectorizer(config: Dict[str, Any]):
    """Create vectorizer instance from config using class name.

    The vectorizer config should have:
    - class: The exact class name (e.g., "OpenAITextVectorizer")
    - model: The model name
    - Any additional kwargs are passed to the constructor
    """
    vec_config = config.get("vectorizer", {}).copy()

    class_name = vec_config.pop("class", None)
    if not class_name:
        raise ValueError("Vectorizer 'class' is required in configuration")

    # Import the vectorizer class dynamically
    import redisvl.utils.vectorize as vectorize_module

    if not hasattr(vectorize_module, class_name):
        raise ValueError(
            f"Unknown vectorizer class: {class_name}. "
            f"Must be a class from redisvl.utils.vectorize"
        )

    vectorizer_class = getattr(vectorize_module, class_name)

    # All remaining config keys are passed as kwargs to the constructor
    return vectorizer_class(**vec_config)
```

### Settings Class

```python
# redisvl/mcp/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class MCPSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REDISVL_MCP_",
        env_file=".env",
        extra="ignore",
    )

    # Path to unified MCP configuration file
    config: str  # Required: path to mcp_config.yaml

    # Server mode (can also be set in config file, env var takes precedence)
    read_only: bool = False

    # Tool descriptions (customizable for agent context)
    tool_search_description: str = (
        "Search for records in the Redis vector database. "
        "Supports semantic search, full-text search, and hybrid search."
    )
    tool_upsert_description: str = (
        "Upsert records into the Redis vector database. "
        "Records are automatically embedded and indexed."
    )
```

---

## Tools

### Tool: `redisvl-search`

Search for records using vector similarity, full-text, or hybrid search.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str | Yes | The search query text |
| `search_type` | str | No | One of: `vector`, `fulltext`, `hybrid`. Default: `vector` |
| `limit` | int | No | Maximum results to return. Default: 10 |
| `offset` | int | No | Pagination offset. Default: 0 |
| `filter` | dict | No | Filter expression (field conditions) |
| `return_fields` | list[str] | No | Fields to return. Default: all fields |

#### Implementation

```python
# redisvl/mcp/tools/search.py
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import Context

async def search(
    ctx: Context,
    query: str,
    search_type: str = "vector",
    limit: int = 10,
    offset: int = 0,
    filter: Optional[Dict[str, Any]] = None,
    return_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search for records in the Redis vector database."""
    server = ctx.server  # RedisVLMCPServer instance
    index = server.index
    vectorizer = server.vectorizer

    if search_type == "vector":
        # Generate embedding for query (as_buffer=True for efficient query integration)
        embedding = await vectorizer.aembed(query, as_buffer=True)

        # Build VectorQuery
        from redisvl.query import VectorQuery
        q = VectorQuery(
            vector=embedding,
            vector_field_name=server.vector_field_name,
            num_results=limit,
            return_fields=return_fields,
        )
        if filter:
            q.set_filter(build_filter_expression(filter))

    elif search_type == "fulltext":
        from redisvl.query import TextQuery
        q = TextQuery(
            text=query,
            text_field_name=server.text_field_name,
            num_results=limit,
            return_fields=return_fields,
        )
        if filter:
            q.set_filter(build_filter_expression(filter))

    elif search_type == "hybrid":
        # Generate embedding for query (as_buffer=True for efficient query integration)
        embedding = await vectorizer.aembed(query, as_buffer=True)
        from redisvl.query import HybridQuery
        q = HybridQuery(
            text=query,
            text_field_name=server.text_field_name,
            vector=embedding,
            vector_field_name=server.vector_field_name,
            num_results=limit,
        )
    else:
        raise ValueError(f"Invalid search_type: {search_type}")

    # Execute query with pagination
    q.paging(offset, limit)
    results = await index.query(q)

    return results
```

#### Response Format

Returns a list of matching records:

```json
[
  {
    "id": "doc:123",
    "score": 0.95,
    "content": "The document text...",
    "metadata_field": "value"
  }
]
```

---

### Tool: `redisvl-upsert`

Upsert records into the index. This tool is **excluded when `read_only=true`**.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `records` | list[dict] | Yes | Records to upsert |
| `id_field` | str | No | Field to use as document ID |
| `embed_field` | str | No | Field containing text to embed. Default: auto-detect |

#### Implementation

```python
# redisvl/mcp/tools/upsert.py
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import Context

async def upsert(
    ctx: Context,
    records: List[Dict[str, Any]],
    id_field: Optional[str] = None,
    embed_field: Optional[str] = None,
) -> Dict[str, Any]:
    """Upsert records into the Redis vector database."""
    server = ctx.server
    index = server.index
    vectorizer = server.vectorizer

    # Determine which field to embed
    if embed_field is None:
        embed_field = server.default_embed_field

    # Generate embeddings for all records (as_buffer=True for storage efficiency)
    texts_to_embed = [record.get(embed_field, "") for record in records]
    embeddings = await vectorizer.aembed_many(texts_to_embed, as_buffer=True)

    # Add embeddings to records (already in buffer format for Redis storage)
    vector_field = server.vector_field_name
    for record, embedding in zip(records, embeddings):
        record[vector_field] = embedding

    # Load records into index
    keys = await index.load(
        data=records,
        id_field=id_field,
    )

    return {
        "status": "success",
        "keys_upserted": len(keys),
        "keys": keys,
    }
```

#### Response Format

```json
{
  "status": "success",
  "keys_upserted": 3,
  "keys": ["doc:abc123", "doc:def456", "doc:ghi789"]
}
```

---

## Server Implementation

### RedisVLMCPServer Class

```python
# redisvl/mcp/server.py
from mcp.server.fastmcp import FastMCP
from redisvl.index import AsyncSearchIndex
from redisvl.mcp.settings import MCPSettings
from redisvl.mcp.config import load_mcp_config, create_index_schema, create_vectorizer

class RedisVLMCPServer(FastMCP):
    """MCP Server for RedisVL vector database operations."""

    def __init__(self, settings: MCPSettings | None = None):
        self.settings = settings or MCPSettings()
        super().__init__(name="redisvl")

        # Load unified configuration
        self._config = load_mcp_config(self.settings.config)

        # Initialize index and vectorizer lazily
        self._index: AsyncSearchIndex | None = None
        self._vectorizer = None

        # Register tools
        self._setup_tools()

    async def _get_index(self) -> AsyncSearchIndex:
        """Lazy initialization of the search index."""
        if self._index is None:
            schema = create_index_schema(self._config)
            redis_url = self._config.get("redis_url", "redis://localhost:6379")
            self._index = AsyncSearchIndex(
                schema=schema,
                redis_url=redis_url,
            )
        return self._index

    async def _get_vectorizer(self):
        """Lazy initialization of the vectorizer."""
        if self._vectorizer is None:
            self._vectorizer = create_vectorizer(self._config)
        return self._vectorizer

    def _setup_tools(self):
        """Register MCP tools."""
        from redisvl.mcp.tools.search import search

        # Always register search tool
        self.tool(
            search,
            name="redisvl-search",
            description=self.settings.tool_search_description,
        )

        # Conditionally register upsert tool
        if not self.settings.read_only:
            from redisvl.mcp.tools.upsert import upsert
            self.tool(
                upsert,
                name="redisvl-upsert",
                description=self.settings.tool_upsert_description,
            )

    @property
    def index(self) -> AsyncSearchIndex:
        """Access the search index (for tool implementations)."""
        # Note: Tools should use await self._get_index() for lazy init
        return self._index

    @property
    def vectorizer(self):
        """Access the vectorizer (for tool implementations)."""
        return self._vectorizer
```

---

## CLI Integration

### Command Structure

```bash
# Start MCP server (stdio transport) - requires config file
rvl mcp --config path/to/mcp_config.yaml

# Read-only mode (overrides config file setting)
rvl mcp --config path/to/mcp_config.yaml --read-only
```

### Implementation

```python
# redisvl/cli/mcp.py
import argparse
import sys

def add_mcp_parser(subparsers):
    """Add MCP subcommand to CLI."""
    parser = subparsers.add_parser(
        "mcp",
        help="Start the RedisVL MCP server",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to MCP configuration YAML file (overrides REDISVL_MCP_CONFIG)",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Run in read-only mode (no upsert tool)",
    )
    parser.set_defaults(func=run_mcp_server)

def run_mcp_server(args):
    """Run the MCP server."""
    try:
        from redisvl.mcp import RedisVLMCPServer, MCPSettings
    except ImportError:
        print(
            "MCP dependencies not installed. "
            "Install with: pip install redisvl[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build settings from args + environment
    settings_kwargs = {}
    if args.config:
        settings_kwargs["config"] = args.config
    if args.read_only:
        settings_kwargs["read_only"] = True

    settings = MCPSettings(**settings_kwargs)
    server = RedisVLMCPServer(settings=settings)

    # Run with stdio transport
    server.run(transport="stdio")
```

### Integration with Existing CLI

Modify `redisvl/cli/main.py` to add the MCP subcommand:

```python
# In create_parser() or equivalent
from redisvl.cli.mcp import add_mcp_parser
add_mcp_parser(subparsers)
```

---

## Client Configuration Examples

### Claude Desktop

```json
{
  "mcpServers": {
    "redisvl": {
      "command": "uvx",
      "args": ["--from", "redisvl[mcp]", "rvl", "mcp", "--config", "/path/to/mcp_config.yaml"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Alternatively, use the environment variable for the config path:

```json
{
  "mcpServers": {
    "redisvl": {
      "command": "uvx",
      "args": ["--from", "redisvl[mcp]", "rvl", "mcp"],
      "env": {
        "REDISVL_MCP_CONFIG": "/path/to/mcp_config.yaml",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Claude Agents SDK (Python)

```python
import os
from agents import Agent
from agents.mcp import MCPServerStdio

async def main():
    async with MCPServerStdio(
        command="uvx",
        args=["--from", "redisvl[mcp]", "rvl", "mcp", "--config", "mcp_config.yaml"],
        env={
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        },
    ) as server:
        agent = Agent(
            name="search-agent",
            instructions="You help users search the knowledge base.",
            mcp_servers=[server],
        )
        # Use agent...
```

---

## Deliverables Mapping

This specification maps to the project deliverables as follows:

| Deliverable | Specification Section | LOE |
|-------------|----------------------|-----|
| MCP Server Framework in RedisVL | Server Implementation, Architecture | M |
| Tool: Search records | Tools > redisvl-search | S |
| Tool: Upsert records | Tools > redisvl-upsert | S |
| MCP runnable from CLI | CLI Integration | S |
| Integration: Claude Agents SDK | Client Configuration Examples | S |

---

## Implementation Phases

### Phase 1: Core Framework (M)

1. Create `redisvl/mcp/` module structure
2. Implement `MCPSettings` with pydantic-settings
3. Implement `RedisVLMCPServer` extending FastMCP
4. Add `mcp` optional dependency group to pyproject.toml
5. Add basic tests for server initialization

### Phase 2: Search Tool (S)

1. Implement `redisvl-search` tool with vector search
2. Add full-text search support
3. Add hybrid search support
4. Add filter expression parsing
5. Add pagination support
6. Add tests for search functionality

### Phase 3: Upsert Tool (S)

1. Implement `redisvl-upsert` tool
2. Add automatic embedding generation
3. Add read-only mode exclusion logic
4. Add tests for upsert functionality

### Phase 4: CLI Integration (S)

1. Add `mcp` subcommand to CLI
2. Handle optional dependency import gracefully
3. Add CLI argument parsing
4. Test `uvx --from redisvl[mcp] rvl mcp` pattern

### Phase 5: Integration Examples (S)

1. Create Claude Agents SDK example
2. Document Claude Desktop configuration
3. (Bonus) Create ADK example
4. (Bonus) Create n8n workflow example

---

## Testing Strategy

### Unit Tests

Location: `tests/unit/test_mcp/`

- **Settings** (`test_settings.py`)
  - Loading settings from environment variables
  - Default values for optional settings
  - Read-only mode flag handling

- **Configuration** (`test_config.py`)
  - YAML loading and parsing
  - Environment variable substitution (`${VAR}` syntax)
  - IndexSchema creation from config
  - Vectorizer instantiation from class name
  - Error handling for missing/invalid config

### Integration Tests

Location: `tests/integration/test_mcp/`

Requires: Redis instance (use testcontainers)

- **Server initialization** (`test_server.py`)
  - Server starts with valid config
  - Index connection established
  - Tools registered correctly
  - Read-only mode excludes upsert tool

- **Search tool** (`test_search.py`)
  - Vector search returns relevant results
  - Full-text search works correctly
  - Hybrid search combines both methods
  - Pagination (offset/limit) works
  - Filter expressions applied correctly

- **Upsert tool** (`test_upsert.py`)
  - Records inserted into Redis
  - Embeddings generated and stored
  - ID field used for key generation
  - Records retrievable after upsert

---

## Future Considerations

### Additional Transport Protocols

The current implementation supports only `stdio`. Future iterations may add:

- **SSE (Server-Sent Events)**: For remote client connections
- **Streamable HTTP**: For web-based integrations

### Additional Tools

Future tools to consider:

- `redisvl-delete`: Delete records by ID or filter
- `redisvl-count`: Count records matching a filter
- `redisvl-info`: Get index information and statistics
- `redisvl-aggregate`: Run aggregation queries

### Multi-Index Support

The current design supports a single index. Future iterations may support:

- Multiple indexes via configuration
- Dynamic index selection in tool parameters

