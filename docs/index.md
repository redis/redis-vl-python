---
myst:
  html_meta:
    "description lang=en": |
      RedisVL documentation - the AI-native Python client for Redis.
html_theme.sidebar_secondary.remove: false
---

# RedisVL

**The AI-native Python client for Redis.**

RedisVL is the production-ready Python client for AI applications built on Redis. Lightning-fast vector search meets enterprise-grade reliability.

Build **RAG pipelines** with real-time retrieval, **AI agents** with memory and semantic routing, and **recommendation systems** with personalized search and reranking.

---

## Quick Start

```bash
pip install redisvl
```

```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

Or connect to [Redis Cloud](https://redis.io/cloud) for a managed experience.

→ *{doc}`/user_guide/01_getting_started`*

---

## Explore the Docs

::::{grid} 2
:gutter: 4

:::{grid-item-card} 📖 Concepts
:link: concepts/index
:link-type: doc
:class-card: sd-shadow-sm

Understand how RedisVL works. Architecture, search fundamentals, and extension patterns.
:::

:::{grid-item-card} 🚀 User Guides
:link: user_guide/index
:link-type: doc
:class-card: sd-shadow-sm

Step-by-step tutorials. Installation, getting started, and deep dives on every feature.
:::

:::{grid-item-card} 📚 API Reference
:link: api/index
:link-type: doc
:class-card: sd-shadow-sm

Complete API documentation. Classes, methods, parameters, and examples.
:::

:::{grid-item-card} 💡 Examples
:link: examples/index
:link-type: doc
:class-card: sd-shadow-sm

Real-world applications. RAG pipelines, chatbots, recommendation systems, and more.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

Concepts <concepts/index>
User Guides <user_guide/index>
API <api/index>
Examples <examples/index>
Changelog <https://github.com/redis/redis-vl-python/releases>
```
