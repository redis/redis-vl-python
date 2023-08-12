---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for RedisVL, with links to the rest
      of the site..
html_theme.sidebar_secondary.remove: false
---

# Redis Vector Library (RedisVL)

RedisVL provides a powerful Python client library for using Redis as a Vector Database.
Leverage the speed and reliability of Redis along with vector-based semantic search capabilities
to supercharge your application!

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

## Concepts (Coming Soon)

Glossary of AI terminology and concepts related to vector databases.


## User Guide

How to use RedisVL

```{toctree}
:maxdepth: 2

user_guide/index
```


## Examples

Examples of RedisVL

```{toctree}
:maxdepth: 2

examples/index
```


## API

The redisVL API

```{toctree}
:maxdepth: 2

API <api/index>
```


```{toctree}
:hidden:

Changelog <https://github.com/RedisVentures/redisvl/releases>
```