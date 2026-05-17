---
description: Exact build, test, and docs commands for AI agents working on the redisvl repository.
---

# Build and test

The shortest path from a fresh checkout to a green build.

## One-time setup

```bash
make install        # uv sync --all-extras
make redis-start    # docker run -d --name redis -p 6379:6379 redis:8.4
```

`make install` uses [uv](https://docs.astral.sh/uv/) and writes the lock file
in `uv.lock`. Do **not** edit `pyproject.toml` to add or remove dependencies;
use `uv add` / `uv remove` (with `--group dev` or `--group docs` where
appropriate) so the lock stays consistent.

`make redis-start` is required for any test that talks to Redis.
`testcontainers` will create its own ephemeral container per integration test,
but the notebook suite uses the long-lived container on port 6379.

## Format and lint

```bash
make format         # isort + black on redisvl/ and tests/
make check-types    # mypy
make lint           # format + check-types
```

Run `make format` before committing. CI re-runs the same commands.

## Tests

```bash
make test                       # default suite (no API-dependent tests)
make test ARGS="--run-api-tests"  # full suite including paid providers
make test-all                   # alias for the above
make test-verbose               # -vv -s
make test-notebooks             # nbval-lax across docs/user_guide/*.ipynb
```

`make test` runs `pytest -n auto`. Set `ARGS="-k name_substring"` to subset.

API-dependent tests need the appropriate environment variables set, for
example `OPENAI_API_KEY`, `COHERE_API_KEY`, `VOYAGE_API_KEY`,
`LANGCACHE_API_KEY`, etc. Tests that require keys you have not provided are
skipped, not failed.

## Docs

```bash
make docs-build     # mkdocs build --strict
make docs-serve     # mkdocs serve --dev-addr 127.0.0.1:8000
```

`docs-build` is the contract: it must pass with `--strict`, which fails on
broken cross-references and on unrecognized config. `docs-serve` watches the
`redisvl/` package as well as the docs tree, so changes to docstrings reflow
the API reference live.

The docs depend on the `docs` group:

```bash
uv sync --group docs
```

The lock file already pins compatible versions; `make install` pulls these in
as part of the default groups.

## Pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

The hooks run formatting, type checking, and a codespell pass.

## Releasing

The release workflow auto-bumps `pyproject.toml` `version`. **Do not** edit
that field by hand. Tag conventions and the actual cut are handled in CI.

## Things that *will* break the build

- Adding a public symbol without re-exporting it from the relevant
  `__init__.py`. The mkdocstrings pages reference subpackage paths, not module
  paths.
- Changing a docstring section header from a Google-style `Args:` /
  `Returns:` / `Raises:` to a non-Google format. mkdocstrings is configured
  with `docstring_style: google`.
- Renaming a notebook under `docs/user_guide/`. The path appears in
  `mkdocs.yml`'s `nav:` section and in the how-to-guide and use-case index
  cards. Update all three.
- Editing `pyproject.toml` directly to bump or add a dependency. Use `uv add`.
- Adding a doc file outside the `nav:` tree without listing it. `mkdocs build
  --strict` fails on orphaned pages.
