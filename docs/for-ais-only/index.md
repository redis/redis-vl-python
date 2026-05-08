---
description: For AI agents working on the redisvl codebase. Repository map, build and test, failure modes.
---

# For AI Agents

This section is written for AI coding agents working on the redisvl codebase
itself, not for end users of the library.

If you are an AI agent contributing patches, start here. The user-facing docs
(Concepts, User Guide, API Reference) are the right reading for understanding
what the library does. The pages in this section are about *how the source is
laid out*, *how to build and test it*, and *how it tends to fail*.

A flat [`AGENTS.md`](https://github.com/redis/redis-vl-python/blob/main/AGENTS.md)
file lives at the repo root with a usage-oriented quick reference for agents
that need to call the library, not contribute to it.

<div class="grid cards" markdown>

-   :material-map:{ .lg .middle } **[Repository map](REPOSITORY_MAP.md)**

    ---

    Where every package lives, what it owns, and which tests cover it.

-   :material-hammer-wrench:{ .lg .middle } **[Build and test](BUILD_AND_TEST.md)**

    ---

    The exact `make` targets, environment variables, and Redis fixtures the
    test suite expects.

-   :material-bug:{ .lg .middle } **[Failure modes](FAILURE_MODES.md)**

    ---

    The common ways changes break the test suite or surface as runtime errors,
    and the canonical fixes.

</div>

## Machine-readable indexes

- [`llms.txt`](https://docs.redisvl.com/llms.txt) — short, flat index of every doc page in this site.
- [`llms-full.txt`](https://docs.redisvl.com/llms-full.txt) — full content of every doc page concatenated for one-shot loading.
