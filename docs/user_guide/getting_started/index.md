# Getting Started with RedisVL

Welcome to RedisVL! This section will help you get up and running quickly with Redis as your vector database for AI applications.

## 30 Second Overview

RedisVL is a Python library that transforms Redis into a powerful vector database for AI applications:

```python
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

# Create an index
index = SearchIndex.from_yaml("schema.yaml")
index.create()

# Load data
index.load(data)

# Search with vectors
query = VectorQuery(vector=embedding, return_fields=["content"])
results = index.query(query)
```

That's it! You now have a production-ready vector database.

## What is RedisVL?

RedisVL is a Python library that makes it easy to use Redis as a vector database for:

| Capability | Description |
|------------|-------------|
| **Semantic Search** | Find similar documents, images, or other data by meaning |
| **RAG Applications** | Build AI applications that combine LLMs with your data |
| **LLM Caching** | Cache LLM responses semantically to reduce costs 10-100x |
| **Query Routing** | Route user queries to appropriate handlers by intent |
| **Chatbot Memory** | Store and retrieve conversation history intelligently |

## Prerequisites

::::{grid} 3
:gutter: 2

:::{grid-item-card} Python
Python 3.9 or higher
:::

:::{grid-item-card} Redis
Redis Stack or Redis Cloud with Search module
:::

:::{grid-item-card} Knowledge
Basic Python and understanding of embeddings
:::

::::

### Setting Up Redis

**Option 1: Docker (Recommended for development)**
```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

**Option 2: Redis Cloud (Recommended for production)**
Sign up for free at [Redis Cloud](https://redis.io/cloud) - includes Search module.

**Option 3: Local Installation**
Follow the [Redis Stack installation guide](https://redis.io/docs/install/install-stack/).

### Installing RedisVL

```bash
pip install redisvl
```

Verify the installation:
```bash
rvl version
```

## Quick Start Guide

The fastest way to get started is with our [Getting Started notebook](../01_getting_started.ipynb), which covers:

1. **Define a Schema** - Structure your data with `IndexSchema`
2. **Create an Index** - Set up your vector index with `SearchIndex`
3. **Load Data** - Ingest your documents with embeddings
4. **Query** - Perform semantic search with `VectorQuery`
5. **Filter** - Combine vector search with metadata filters

## Core Concepts

### Understanding the Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                       RedisVL                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Schema  │  │  Index   │  │  Query   │  │Extensions│ │
│  │          │  │          │  │          │  │(Cache,  │ │
│  │ Define   │→ │ Create   │→ │ Search   │  │ Router) │ │
│  │ Structure│  │ & Load   │  │ & Filter │  │         │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     Redis Stack                          │
│         (Vector Search + Full-Text + JSON + More)        │
└─────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Example Use |
|-----------|---------|-------------|
| **IndexSchema** | Defines structure of your data | Field types, vector dimensions, distance metric |
| **SearchIndex** | Manages your Redis index | Create, load, update, delete operations |
| **VectorQuery** | Performs similarity searches | Find K nearest neighbors |
| **FilterQuery** | Filters by metadata | Filter by category, date, price |
| **Vectorizers** | Generate embeddings | OpenAI, Cohere, HuggingFace integrations |

## Learning Path

We recommend following this progression:

```
                    ┌─────────────────┐
                    │  Getting Started │ (15 min)
                    │   This guide!    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Semantic Cache  │ │ Message History │ │ Semantic Router │
│   (20 min)      │ │    (25 min)     │ │    (30 min)     │
│ Cache LLM calls │ │ Build chatbots  │ │ Route by intent │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  How-To Guides  │
                    │  Advanced Topics │
                    └─────────────────┘
```

### Suggested Timeline

| Phase | Duration | Topics |
|-------|----------|--------|
| **Day 1** | 30 min | Getting Started + Semantic Caching |
| **Day 2** | 30 min | Message History + Vectorizers |
| **Week 1** | 2 hrs | Advanced Queries + Reranking + Optimization |

## Next Steps

::::{grid} 2
:gutter: 2

:::{grid-item-card} → Tutorials
Learn features step-by-step with hands-on examples.

[Go to Tutorials](../tutorials/index)
:::

:::{grid-item-card} → How-To Guides
Find solutions for specific tasks and problems.

[Go to How-To Guides](../how_to_guides/index)
:::

:::{grid-item-card} → Use Cases
See complete application examples.

[Go to Use Cases](../use_cases/index)
:::

:::{grid-item-card} → API Reference
Dive deep into the API details.

[Go to API Docs](/api/index)
:::

::::

## FAQ

:::{dropdown} Do I need a Redis Cloud account?
:icon: question
:animate: fade-in

No, you can use a local Redis instance with Docker or any Redis deployment with the Search module enabled.
:::

:::{dropdown} What embedding models can I use?
:icon: question
:animate: fade-in

RedisVL works with any embedding model! We provide built-in support for:
- OpenAI (text-embedding-3-small, text-embedding-3-large)
- Cohere (embed-v3)
- HuggingFace models
- Voyage AI
- Custom embeddings

See our [Vectorizers guide](../04_vectorizers.ipynb) for setup instructions.
:::

:::{dropdown} Can I use RedisVL with LangChain or LlamaIndex?
:icon: question
:animate: fade-in

Yes! RedisVL integrates seamlessly with popular LLM frameworks. You can use Redis as:
- A vector store in LangChain
- An index in LlamaIndex
- A standalone vector database for any framework
:::

:::{dropdown} How does RedisVL compare to other vector databases?
:icon: question
:animate: fade-in

RedisVL offers unique advantages:
- **Speed**: Sub-millisecond latency with in-memory storage
- **Simplicity**: Single deployment for vectors, caching, and more
- **Flexibility**: Combine vector search with full-text, JSON, and filtering
- **Battle-tested**: Redis is used by millions of applications worldwide
:::

## Need Help?

- **📖 Documentation**: Use the search bar to find specific topics
- **🐛 GitHub Issues**: [Report bugs](https://github.com/redis/redis-vl-python/issues) or request features
- **💬 Community**: Join our [Discord](https://discord.gg/redis) for discussions
- **📧 Support**: Contact Redis support for enterprise needs

