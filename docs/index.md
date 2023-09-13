---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for RedisVL, with links to the rest
      of the site..
html_theme.sidebar_secondary.remove: false
---

# Redis Vector Library (RedisVL)

RedisVL provides a powerful, dedicated Python client library for using Redis as a [Vector Database](https://redis.com/solutions/use-cases/vector-database). Leverage the speed and reliability of Redis along with vector-based semantic search capabilities to supercharge your application!

```{gallery-grid}
:grid-columns: 1 2 2 3

- header: "{fab}`bootstrap;pst-color-primary` Index Management"
  content: "Manipulate Redis search indices in Python or from CLI."
- header: "{fas}`bolt;pst-color-primary` Vector Similarity Search"
  content: "Perform powerful vector similarity search with filtering support."
- header: "{fas}`circle-half-stroke;pst-color-primary` Embedding Creation"
  content: "Use OpenAI or any of the other supported vectorizers to create embeddings."
- header: "{fas}`palette;pst-color-primary` CLI"
  content: "Interact with RedisVL using a Command line interface (CLI) for ease of use."
- header: "{fab}`python;pst-color-primary` Semantic Caching"
  content: "Use RedisVL to cache LLM results, increasing QPS and decreasing cost."
- header: "{fas}`lightbulb;pst-color-primary` Example Gallery"
  content: "Explore our gallery of examples to get started."
  link: "examples/index"
```

## Installation

Install `redisvl` into your Python (>=3.8) environment using `pip`:

```bash
pip install redisvl
```

Then make sure to have [Redis](https://redis.io) accessible with Search & Query features enabled on [Redis Cloud](https://redis.com/try-free) or locally in docker with [Redis Stack](https://redis.io/docs/getting-started/install-stack/docker/):

```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

This will also spin up the [Redis Insight GUI](https://redis.com/redis-enterprise/redis-insight/) at `http://localhost:8001`.

## Table of Contents

```{toctree}
:maxdepth: 2

User Guides <user_guide/index>
Example Gallery <examples/index>
API <api/index>
Concepts Library <concepts/index>

```
