---
description: Source-tree map of the redisvl repository for AI agents working on the codebase.
---

# Repository map

The redisvl repo is a flat Python package plus the docs and tests. Every public
symbol is reachable from `redisvl.<subpackage>`; sub-modules under those names
are implementation detail and may be reorganized between minor versions.

## Top level

| Path | Purpose |
|------|---------|
| `redisvl/` | The library source. |
| `tests/` | Pytest suite. `unit/` is offline; `integration/` requires Redis (via `testcontainers`); `notebooks/` runs the user-guide notebooks. |
| `docs/` | mkdocs source tree (Concepts, User Guide, Examples, API, For AI Agents). |
| `mkdocs.yml` | Site config (Material theme, Redis branding, mkdocstrings, mkdocs-jupyter, llmstxt). |
| `pyproject.toml` | Package metadata, optional extras, dev/docs dep groups. |
| `Makefile` | Convenience targets for install, format, lint, test, docs. |
| `.readthedocs.yaml` | RTD build config (uv install + mkdocs build). |
| `AGENTS.md` | Quick reference for AI agents *using* the library. |
| `CLAUDE.md` | Project conventions surfaced to coding agents. |
| `CONTRIBUTING.md` | Contributor onboarding. |

## Package layout

```
redisvl/
├── __init__.py            # version + public re-exports
├── exceptions.py          # public exception classes
├── types.py               # type aliases for sync/async redis clients
├── version.py             # __version__ source of truth
│
├── cli/                   # `rvl` CLI entry points
│   ├── runner.py          # main(); wires sub-commands
│   ├── index.py           # `rvl index` (create, list, info, delete)
│   ├── stats.py           # `rvl stats`
│   ├── mcp.py             # `rvl mcp`
│   └── version.py         # `rvl version`
│
├── extensions/
│   ├── cache/             # SemanticCache, EmbeddingsCache, LangCacheSemanticCache
│   ├── llmcache/          # legacy alias kept for back-compat imports
│   ├── message_history/   # MessageHistory + SemanticMessageHistory
│   ├── router/            # SemanticRouter, Route, RoutingConfig
│   └── session_manager/   # deprecated alias of message_history
│
├── index/
│   ├── index.py           # SearchIndex, AsyncSearchIndex
│   └── storage.py         # Hash and JSON storage adapters
│
├── mcp/                   # `rvl mcp` server implementation
│   ├── config.py          # YAML config loading + env-var substitution
│   ├── server.py          # FastMCP server wiring
│   ├── tools/             # search + upsert MCP tool implementations
│   └── filters.py         # schema-derived typed filter helpers
│
├── migration/             # internal migration helpers
│
├── query/
│   ├── query.py           # VectorQuery, VectorRangeQuery, FilterQuery, CountQuery, TextQuery, MultiVectorQuery, AggregateHybridQuery, Vector
│   ├── hybrid.py          # HybridQuery (FT.SEARCH-based hybrid)
│   ├── filter.py          # Tag/Text/Num/Geo/GeoRadius filter expressions
│   ├── aggregate.py       # AggregateQuery base class
│   └── sql.py             # SQLQuery (delegates to sql-redis)
│
├── redis/
│   ├── connection.py      # client factories, sentinel, cluster helpers
│   ├── constants.py       # version constants
│   └── utils.py           # bytes/dict conversions, cluster-aware FT helpers
│
├── schema/
│   ├── schema.py          # IndexSchema, IndexInfo, StorageType
│   ├── fields.py          # TextField, TagField, NumericField, GeoField, FlatVectorField, HNSWVectorField, SVSVectorField + *Attributes
│   ├── type_utils.py
│   └── validation.py
│
└── utils/
    ├── compression.py     # CompressionAdvisor, SVSConfig
    ├── full_text_query_helper.py
    ├── log.py
    ├── redis_protocol.py
    ├── token_escaper.py
    ├── utils.py           # deprecated_argument, deprecated_function, sync_wrapper
    ├── rerank/            # CohereReranker, HFCrossEncoderReranker, VoyageAIReranker
    └── vectorize/         # text/{openai,azureopenai,cohere,huggingface,mistral}, voyageai, vertexai, bedrock, custom
```

## Where things are publicly imported from

User-facing imports use the **subpackage**, never the module:

```python
from redisvl.index import SearchIndex, AsyncSearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery, FilterQuery, HybridQuery, MultiVectorQuery, Vector
from redisvl.query.filter import Tag, Text, Num, Geo, GeoRadius
from redisvl.extensions.cache.llm import SemanticCache, LangCacheSemanticCache
from redisvl.extensions.message_history import MessageHistory, SemanticMessageHistory
from redisvl.extensions.router import SemanticRouter, Route, RoutingConfig
from redisvl.utils.vectorize import HFTextVectorizer, OpenAITextVectorizer  # etc.
from redisvl.utils.rerank import CohereReranker, HFCrossEncoderReranker, VoyageAIReranker
```

If you change the **module** something lives in, keep the subpackage import
working: re-export through `__init__.py`.

## Test layout

| Path | Scope |
|------|-------|
| `tests/unit/` | Pure unit tests, no Redis. Safe to run anywhere. |
| `tests/integration/` | Spins up Redis via `testcontainers`. Requires Docker. |
| `tests/notebooks/` (via `make test-notebooks`) | Executes every `docs/user_guide/*.ipynb` end-to-end with `nbval-lax`. |

API-dependent tests (LangCache, paid vectorizer/reranker providers) only run
when invoked with `pytest --run-api-tests`. They require the relevant API keys.
