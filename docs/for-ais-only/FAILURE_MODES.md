---
description: Common ways changes break the redisvl test suite or surface as runtime errors, and the canonical fixes.
---

# Failure modes

The recurring ways a redisvl change goes wrong, and the smallest fix that
restores green.

## `mkdocs build --strict` fails

### "Doc file ... contains a link to ... which is not found"

A page links to another page that doesn't exist (typo, wrong relative path,
or the target file was deleted/renamed).

- **Notebook references**: notebook nodes are addressed by their on-disk
  path, including the `.ipynb` suffix. From `docs/concepts/extensions.md` the
  Getting Started notebook is `../user_guide/01_getting_started.ipynb`, **not**
  `../user_guide/01_getting_started.md`.
- **API references**: API pages use `.md`. From `docs/concepts/utilities.md`
  the vectorizer reference is `../api/vectorizer.md`.
- **Concept-to-concept refs**: stay flat — `[Architecture](architecture.md)`,
  not `[Architecture](../concepts/architecture.md)`.

### "404 page exists but is not in the navigation"

You added a new page under `docs/` but didn't add it to `nav:` in
`mkdocs.yml`. Either add it or delete it.

### "mkdocstrings could not find ..."

The dotted path under `:::` does not import. mkdocstrings imports the package
in-process; if the import fails (missing optional extra, circular import) the
whole API page fails. Reproduce with:

```bash
uv run python -c "from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer"
```

If it fails, the docs will fail too. Either fix the import or wrap the page's
listing in a different module path.

## Tests fail with `redis.exceptions.ConnectionError`

The container is not running:

```bash
make redis-start
```

If `make redis-start` says `Conflict. The container name "/redis" is already
in use`, you have a stopped container; `make redis-stop && make redis-start`.

## Notebook tests (`make test-notebooks`) fail with output mismatch

`nbval-lax` compares cell outputs loosely, but it still flags hard
exceptions and changed sentinel values. Common causes:

- Schema or query API changed and the notebook still calls the old surface —
  re-run the notebook locally, save the corrected outputs, commit.
- A newer Redis version changed default behavior (most often around stopwords
  or scoring). Either pin the notebook to a Redis version or rewrite the
  assertion.
- The SVS-VAMANA notebook (`09_svs_vamana.ipynb`) requires Redis 8.2+. The
  Makefile auto-skips it when running against an older container.

## Type checking fails after adding an optional dependency

Optional providers (OpenAI, Cohere, etc.) are imported lazily inside the
vectorizer modules. If you add a new top-level import for an optional package,
either:

- gate it behind `if TYPE_CHECKING:` for type-only imports, or
- wrap it in a try/except `ImportError` and raise a clear error from
  `__init__` when the provider is actually used.

Mypy is configured with `ignore_missing_imports = true`, so you will see the
failure first as a runtime `ImportError`, not a type error.

## `uv sync` complains about a stale lock

Someone hand-edited `pyproject.toml`. Re-run the dependency change through uv:

```bash
uv add <pkg>             # or
uv add --group docs <pkg>
```

That regenerates `uv.lock` consistently.

## A notebook drifts from the docs

Three things must agree on every notebook rename or addition:

1. The notebook file under `docs/user_guide/`.
2. The `nav:` entry in `mkdocs.yml`.
3. The card and quick-reference table entries in
   `docs/user_guide/how_to_guides/index.md` and
   `docs/user_guide/use_cases/index.md`.

If you add a notebook and only do (1) + (2), the user-facing index pages will
silently miss the new content; `mkdocs build --strict` will not flag that.
