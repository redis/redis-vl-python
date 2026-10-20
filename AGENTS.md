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

Run the pre-PR review and put its Reviewer Brief in the PR description. A human approves every merge to `main`. The review catches problems before they reach that person, and the brief gives them a head start: where to begin reading, where the change could still fail, and what nobody has checked. Settle any open questions with the author before the PR is opened or marked ready, so the PR reaches the reviewer with none.

The procedure lives in `.agents/skills/pre-pr-review/SKILL.md`, in the Agent Skills format. If your agent doesn't discover skills there by itself, open `SKILL.md` and follow it as written. It's plain Markdown, and every file it refers to is under `.agents/skills/pre-pr-review/`.

In short, the review:

1. Reads the full diff against `main`, plus the callers of anything public that changed.
2. Runs six review lenses (correctness, security, design, tests, docs and Redis behaviour), each starting fresh from the diff. Authors who want to save tokens can run fewer; correctness always runs.
3. Runs a separate challenger pass that drops any finding it can't tie to a file, a line and a concrete failure.
4. Fixes the surviving findings, in scope, and re-runs the checks those fixes touched.
5. Writes the Reviewer Brief into the PR description. `scripts/update_brief.py` replaces just that section.

Write the brief for a reviewer who sees the change as it is now and has never seen your review. Point them at the current code, not at problems you've already fixed. Re-run the review after any substantive push, because the brief names the commit it reviewed.

## Redis documentation

Check Redis behaviour against the official documentation, not memory. Redis hosts a public, read-only MCP server for its docs at `https://redis.io/mcp`, with no API key or sign-up; setup for each client is at `https://marketing.redis.io/docs-mcp/`. Its `search` and `fetch` tools are the quick way to confirm command syntax, defaults and version-specific behaviour while you work, and the pre-PR review's Redis behaviour lens uses them. Send it questions about Redis, not code or credentials.

## Responding to review

Cursor Bugbot reviews every PR, and humans may comment too. Handle each point as `SKILL.md` describes under "Responding to review": verify it against the code, give it one outcome, reply on the thread, and update the Reviewer Brief if anything changed. Treat a reviewer's wording as a claim to check, not a verdict.

## Honesty rules

- Never claim a check in the Reviewer Brief unless you ran it and saw its output.
- Say what you didn't check. Cluster tests and tests that need API keys don't run in CI, so name them when the change touches those paths.
