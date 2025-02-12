---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for RedisVL, with links to the rest
      of the site..
html_theme.sidebar_secondary.remove: false
---

# Redis Vector Library (RedisVL)

A powerful, AI-native Python client library for [Redis](https://redis.io). Leverage the speed, flexibility, and reliability of Redis for real-time data to supercharge your AI application.

```{gallery-grid}
:grid-columns: 1 2 2 3

- header: "{fab}`bootstrap;pst-color-primary` Index Management"
  content: "Design search schema and indices with ease from YAML, with Python, or from the CLI."
- header: "{fas}`bolt;pst-color-primary` Advanced Vector Search"
  content: "Perform powerful vector search queries with complex filtering support."
- header: "{fas}`circle-half-stroke;pst-color-primary` Embedding Creation"
  content: "Use OpenAI or any of the other supported vectorizers to create embeddings."
  link: "user_guide/04_vectorizers"
- header: "{fas}`palette;pst-color-primary` CLI"
  content: "Interact with RedisVL using a Command Line Interface (CLI) for ease of use."
- header: "{fab}`python;pst-color-primary` Semantic Caching"
  content: "Extend RedisVL to cache LLM results, increasing QPS and decreasing system cost."
  link: "user_guide/03_llmcache"
- header: "{fas}`lightbulb;pst-color-primary` Example Gallery"
  content: "Explore the gallery of examples to get started."
  link: "examples/index"
```

## Installation

Install `redisvl` into your Python (>=3.8) environment using `pip`:

```bash
pip install redisvl
```

Then make sure to have [Redis](https://redis.io) accessible with Search & Query features enabled on [Redis Cloud](https://redis.io/cloud) or locally in docker with [Redis Stack](https://redis.io/docs/getting-started/install-stack/docker/):

```bash
docker run -d --name redis -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

This will also spin up the [Redis Insight GUI](https://redis.io/insight/) at `http://localhost:8001`.


## Table of Contents

```{toctree}
:maxdepth: 2

Overview <overview/index>
API <api/index>
User Guides <user_guide/index>
Example Gallery <examples/index>
```

```{toctree}
:hidden:

Changelog <https://github.com/redis/redis-vl-python/releases>
```
