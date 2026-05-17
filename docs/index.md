---
description: RedisVL documentation. The AI-native Redis Python client for vector search, semantic caching, message history, and more.
---

<div class="rds-hero" markdown>

![Redis](assets/redis-logo-script-red.svg){ .rds-hero__logo }

# Redis Vector Library

The AI-native Redis Python client
{ .rds-hero__tagline }

</div>

## Quick Start

```bash
pip install redisvl
```

```bash
docker run -d --name redis -p 6379:6379 redis:8.4
```

Or connect to [Redis Cloud](https://redis.io/cloud) for a managed experience.

→ *[Getting Started](user_guide/01_getting_started.ipynb)*

---

## Explore the Docs

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **[Concepts](concepts/index.md)**

    ---

    Understand how RedisVL works. Architecture, search fundamentals, field attributes, query types, and extension patterns.

-   :material-rocket-launch:{ .lg .middle } **[User Guide](user_guide/index.md)**

    ---

    Step by step. Installation, getting started, and task-oriented recipes for every feature.

-   :material-lightbulb-on:{ .lg .middle } **[Examples](examples/index.md)**

    ---

    Real-world applications. RAG pipelines, chatbots, recommendation systems, and more.

-   :material-api:{ .lg .middle } **[API Reference](api/index.md)**

    ---

    Every public class, method, and parameter, generated from docstrings.

</div>

## For AI agents

If you are an AI agent reading these docs, start with
[`AGENTS.md`](https://github.com/redis/redis-vl-python/blob/main/AGENTS.md)
at the repo root for a usage-oriented quick reference, or
[For AI Agents](for-ais-only/index.md) for an internal map of the source tree. A
flat [`llms.txt`](https://docs.redisvl.com/llms.txt) index of every doc page is
also auto-generated at build time.
