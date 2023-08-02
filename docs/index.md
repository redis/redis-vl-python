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
  content: "Manipulate Redis index in Python for from a CLI."
- header: "{fas}`bolt;pst-color-primary` Vector Search"
  content: "Simple vector search capabilities supporting synchronous and asyncronous search."
- header: "{fas}`circle-half-stroke;pst-color-primary` Embedding Creation"
  content: "User OpenAI or any of the other supported vectorizers to create embeddings"
- header: "{fas}`palette;pst-color-primary` CLI"
  content: "Command line interface for RedisVL makes interacting with Redis as a vector database easy."
- header: "{fab}`python;pst-color-primary` Semantic Caching"
  content: "Use RedisVL to cache the results of your LLM models increasing QPS and decreasing cost."
- header: "{fas}`lightbulb;pst-color-primary` Example Gallery"
  content: "See our gallery of projects that use RedisVL"
  link: "examples/index"
```



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

## Developer

How to contribute

```{toctree}
:maxdepth: 2

developer/index
```


```{toctree}
:hidden:

Changelog <https://github.com/pydata/pydata-sphinx-theme/releases>
```