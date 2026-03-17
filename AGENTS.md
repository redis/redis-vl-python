# AGENTS.md - RedisVL Project Context

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

### Git Operations
**CRITICAL**: NEVER use `git push` or attempt to push to remote repositories. The user will handle all git push operations.

### Branch and Commit Policy
**IMPORTANT**: Use conventional branch names and conventional commits.

Branch naming:
- Human-created branches should use `<type>/<short-kebab-description>`
- Automation-created branches may use `codex/<type>/<short-kebab-description>`
- Preferred branch types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`
- Examples:
  - `feat/index-migrator`
  - `fix/async-sentinel-pool`
  - `docs/index-migrator-benchmarking`
  - `codex/feat/index-migrator`

Commit messages:
- Use Conventional Commits: `<type>(optional-scope): <summary>`
- Preferred commit types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`
- Examples:
  - `feat(migrate): add drop recreate planning docs`
  - `docs(index-migrator): add benchmarking guidance`
  - `fix(cli): validate migrate plan inputs`

### Code Quality
**IMPORTANT**: Always run `make format` before committing code to ensure proper formatting and linting compliance.

### README.md Maintenance
**IMPORTANT**: DO NOT modify README.md unless explicitly requested.

**If you need to document something, use these alternatives:**
- Development info → CONTRIBUTING.md
- API details → docs/ directory
- Examples → docs/examples/
- Project memory (explicit preferences, directives, etc.) → AGENTS.md

## Code Style Preferences

### Import Organization
- **Prefer module-level imports** by default for clarity and standard Python conventions
- **Use local/inline imports only when necessary** for specific reasons:
  - Avoiding circular import dependencies
  - Improving startup time for heavy/optional dependencies
  - Lazy loading for performance-critical paths
- When using local imports, add a brief comment explaining why (e.g., `# Local import to avoid circular dependency`)

### Comments and Output
- **No emojis in code comments or print statements**
- Keep comments professional and focused on technical clarity
- Use emojis sparingly only in user-facing documentation (markdown files), not in Python code

### General Guidelines
- Follow existing patterns in the RedisVL codebase
- Maintain consistency with the project's established conventions
- Run `make format` before committing to ensure code quality standards

## Testing Notes
RedisVL uses `pytest` with `testcontainers` for testing.

- `make test` - unit tests only (no external APIs)
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
