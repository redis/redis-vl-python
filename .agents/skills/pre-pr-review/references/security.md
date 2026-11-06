# Security lens

Find changes that let input reach somewhere it shouldn't, or let a caller do or see more than they're allowed to. Report only issues this diff introduces or makes reachable, not ones that already existed.

## Look for

- **Query-syntax injection.** A value that leaves its RediSearch clause and changes the query: tag alternation (`|`) widening a tag clause, a quote or backslash breaking out of a text phrase, wildcards or operators passing through unescaped. Treat this as injection, even though it isn't SQL.
- **Asymmetric escaping.** A value escaped at query time but not at index time, or the other way round, either fails to match or matches the wrong documents.
- **Authorisation that fails open.** This is mostly in `redisvl/mcp/`: tenant or claim scoping, locked filters, transport and Host/Origin checks. Look for `getattr(..., None)` chains, empty-list defaults and broad `except` blocks that end in "allow".
- **Destructive operations that hit the wrong target.** `delete`, `clear` and index drops that follow an alias, a prefix or a pattern further than the caller meant them to.
- **Secrets.** API keys, passwords or connection URLs written to logs, exceptions or `repr`.
- **Deserialisation and code execution.** `pickle`, unsafe YAML loading, `eval`, or importing a module named in user-supplied config.
- **CI and supply chain.** Changes under `.github/` that skip tests, widen token permissions, or run untrusted PR content. New or loosened dependency constraints.

## Severity

- **High:** an outsider can read or alter data beyond their scope, or delete data.
- **Medium:** the same, but it needs an unusual configuration or a privileged caller.
- Report nothing below medium from this lens.

## Not in scope

Denial of service through expensive but well-formed queries, rate limiting, and test-only code. Treat environment variables and CLI flags as trusted.
