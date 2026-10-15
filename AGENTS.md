# AGENTS.md

Guidance for coding agents working on RedisVL, whichever agent or model you are.

## Frequently Used Commands

```bash
# Development workflow
make install          # Install dependencies
make format           # Format code (black + isort)
make check-types      # Run mypy type checking
make lint             # Run all linting (format + types)
make test             # Run tests (no external APIs)
make test-all         # Run all tests (includes API tests)
make check            # Full check (lint + test)

# Redis setup
make redis-start      # Start Redis container
make redis-stop       # Stop Redis container

# Documentation
make docs-build       # Build documentation
make docs-serve       # Serve docs locally
```

Pre-commit hooks are also configured, which you should
run before you commit:
```bash
pre-commit run --all-files
```

## Important Architectural Patterns

### Async/Sync Dual Interfaces
- Most core classes have both sync and async versions (e.g., `SearchIndex` / `AsyncSearchIndex`)
- Follow existing patterns when adding new functionality

### Schema-Driven Design
```python
# Index schemas define structure
schema = IndexSchema.from_yaml("schema.yaml")
index = SearchIndex(schema, redis_url="redis://localhost:6379")
```

## Critical Rules

### Do Not Modify
- **CRITICAL**: Do not change this line unless explicitly asked:
  ```python
  token.strip().strip(",").replace(""", "").replace(""", "").lower()
  ```

### Protected Directories
**CRITICAL**: NEVER delete the `local_docs/` directory or any files within it.

### Git Operations
**CRITICAL**: NEVER use `git push` or attempt to push to remote repositories. The user will handle all git push operations.

### Code Quality
**IMPORTANT**: Always run `make format` before committing code to ensure proper formatting and linting compliance.

### README.md Maintenance
**IMPORTANT**: DO NOT modify README.md unless explicitly requested.

**If you need to document something, use these alternatives:**
- Development info → CONTRIBUTING.md
- API details → docs/ directory
- Examples → docs/examples/
- Project memory (explicit preferences, directives, etc.) → AGENTS.md (`CLAUDE.md` only imports this file)

## Testing Notes
RedisVL uses `pytest` with `testcontainers` for testing.

- `make test` - all tests except those that call external APIs
- `make test-all` - run the full suite, including tests that call external APIs
- `pytest --run-api-tests` - explicitly run API-dependent tests (e.g., LangCache,
  external vectorizer/reranker providers). These require the appropriate API
  keys and environment variables to be set.

## Project Structure

```
redisvl/
├── cli/              # Command-line interface (rvl command)
├── extensions/       # AI extensions (cache, memory, routing)
│   ├── cache/        # Semantic caching for LLMs
│   ├── llmcache/     # LLM-specific caching
│   ├── message_history/  # Chat history management
│   ├── router/       # Semantic routing
│   └── session_manager/  # Session management
├── index/            # SearchIndex classes (sync/async)
├── mcp/              # MCP server (tools, auth, transport security)
├── migration/        # Index and data migration
├── query/            # Query builders (Vector, Range, Filter, Count)
├── redis/            # Redis client utilities
├── schema/           # Index schema definitions
└── utils/            # Utilities (vectorizers, rerankers, optimization)
    ├── rerank/       # Result reranking
    └── vectorize/    # Embedding providers integration
```

## Core Components

### 1. Index Management
- `SearchIndex` / `AsyncSearchIndex` - Main interface for Redis vector indices
- `IndexSchema` - Define index structure with fields (text, tags, vectors, etc.)
- Support for JSON and Hash storage types

### 2. Query System
- `VectorQuery` - Semantic similarity search
- `RangeQuery` - Vector search within distance range
- `FilterQuery` - Metadata filtering and full-text search
- `CountQuery` - Count matching records
- Etc.

### 3. AI Extensions
- `SemanticCache` - LLM response caching with semantic similarity
- `EmbeddingsCache` - Cache for vector embeddings
- `MessageHistory` - Chat history with recency/relevancy retrieval
- `SemanticRouter` - Route queries to topics/intents

### 4. Vectorizers (Optional Dependencies)
- OpenAI, Azure OpenAI, Cohere, HuggingFace, Mistral, VoyageAI
- Custom vectorizer support
- Batch processing capabilities

## Documentation
- Main docs: https://docs.redisvl.com
- Built with Sphinx from `docs/` directory
- Includes API reference and user guides
- Example notebooks in documentation `docs/user_guide/...`

## Before you mark a PR ready

Run the pre-PR review and put its Review record in the PR description. A human approves every merge to `main`, and the record is what lets them do that quickly: it tells them which commit was reviewed, from which angles, what was fixed, what was rejected and why, and what was not checked.

The procedure lives in `.agents/skills/pre-pr-review/SKILL.md`, in the Agent Skills format. If your agent doesn't discover skills there by itself, open `SKILL.md` and follow it as written: it's plain Markdown, and every file it refers to is under `.agents/skills/pre-pr-review/references/`.

In short, the review:

1. Reads the full diff against `main`, plus the callers of anything public that changed.
2. Picks which of five lenses the diff needs (correctness always; security, design, tests and docs when the diff calls for them) and records why.
3. Runs one pass per lens, then a separate challenger pass that drops any finding it can't tie to a file, a line and a concrete failure.
4. Fixes the surviving findings, in scope, and re-runs the checks those fixes touched.
5. Writes the Review record into the PR description, replacing any earlier one.

Re-run it after any substantive push. A record that names an older commit than the PR's head is stale.

## Responding to review

Cursor Bugbot reviews every PR, and humans may comment too. Handle each point as `SKILL.md` describes under "Responding to review": verify it against the code, give it one disposition, reply on the thread and add a line to the Review record. Treat a reviewer's wording as a claim to check, not a verdict.

## Honesty rules

- Never list a check under "Verified" unless you ran it and saw its output.
- Say what you didn't check. Cluster tests and tests that need API keys don't run in CI, so name them when the change touches those paths.
