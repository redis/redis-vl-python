---
myst:
  html_meta:
    "description lang=en": |
      User guides for RedisVL - Learn how to build AI applications with Redis as your vector database
---

# Guides

Welcome to the RedisVL guides! Whether you're just getting started or building advanced AI applications, these guides will help you make the most of Redis as your vector database.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📦 Installation
:link: installation
:link-type: doc

**Set up RedisVL.** Install the library and configure your Redis instance for vector search.

+++
pip install • Redis Cloud • Docker
:::

:::{grid-item-card} 🚀 Getting Started
:link: 01_getting_started
:link-type: doc

**New to RedisVL?** Start here to learn the basics and build your first vector search application in minutes.

+++
Schema → Index → Load → Query
:::

:::{grid-item-card} 🛠️ How-To Guides
:link: how_to_guides/index
:link-type: doc

**Solve specific problems.** Task-oriented recipes for LLM extensions, querying, embeddings, optimization, and storage.

+++
LLM Caching • Filtering • Vectorizers • Reranking • Migrations
:::

:::{grid-item-card} 💻 CLI Reference
:link: cli
:link-type: doc

**Command-line tools.** Manage indices, inspect stats, and work with schemas using the `rvl` CLI.

+++
rvl index • rvl stats • rvl migrate • Schema YAML
:::

:::{grid-item-card} 💡 Use Cases
:link: use_cases/index
:link-type: doc

**Apply RedisVL to real-world problems.** See which guides map to your use case.

+++
Agent Context • Agent Optimization • Search • RecSys
:::

::::

```{toctree}
:maxdepth: 2

Installation <installation>
Getting Started <01_getting_started>
how_to_guides/index
CLI Reference <cli>
use_cases/index
```
