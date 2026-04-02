---
myst:
  html_meta:
    "description lang=en": |
      Core concepts for RedisVL - architecture, search, indexing, and AI extensions.
---

# Concepts

Foundational knowledge for building AI applications with RedisVL. These concepts are language-agnostic and apply across all RedisVL implementations.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🏗️ Architecture
:link: architecture
:link-type: doc

How RedisVL components connect: schemas, indexes, queries, and extensions.
:::

:::{grid-item-card} 🔍 Search & Indexing
:link: search-and-indexing
:link-type: doc

Schemas, fields, documents, storage types, and query patterns.
:::

:::{grid-item-card} 🔄 Index Migrations
:link: index-migrations
:link-type: doc

How RedisVL handles migration planning, rebuilds, and future shadow migration.
:::

:::{grid-item-card} 🏷️ Field Attributes
:link: field-attributes
:link-type: doc

Configure sortable, no_index, index_missing, and other field options.
:::

:::{grid-item-card} 🔎 Query Types
:link: queries
:link-type: doc

Vector, filter, text, hybrid, and multi-vector query options.
:::

:::{grid-item-card} 🔧 Utilities
:link: utilities
:link-type: doc

Vectorizers for embeddings and rerankers for result optimization.
:::

:::{grid-item-card} 🧠 MCP
:link: mcp
:link-type: doc

How RedisVL exposes an existing Redis index to MCP clients through a stable tool contract.
:::

:::{grid-item-card} 🧩 Extensions
:link: extensions
:link-type: doc

Pre-built patterns: caching, message history, and semantic routing.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

architecture
search-and-indexing
index-migrations
field-attributes
queries
utilities
mcp
extensions
```
