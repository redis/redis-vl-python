# Getting Started with RedisVL

Welcome to RedisVL! This section will help you get up and running quickly with Redis as your vector database for AI applications.

## What is RedisVL?

RedisVL is a Python library that makes it easy to use Redis as a vector database for:
- **Semantic search**: Find similar documents, images, or other data
- **RAG (Retrieval-Augmented Generation)**: Build AI applications that combine LLMs with your data
- **Recommendation systems**: Power personalized recommendations
- **LLM caching**: Speed up and reduce costs of LLM applications

## Prerequisites

Before you begin, make sure you have:
- Python 3.8 or higher
- A Redis instance (local or cloud) with the Search module enabled
- Basic familiarity with Python and vector embeddings

## Quick Start

The fastest way to get started is with our [Getting Started guide](../01_getting_started.ipynb), which covers:

1. **Installation**: Setting up RedisVL and dependencies
2. **Creating an Index**: Defining your vector search schema
3. **Loading Data**: Ingesting documents into Redis
4. **Searching**: Performing vector similarity searches
5. **Filtering**: Combining vector search with metadata filters

## Learning Path

We recommend following this path:

### 1. Core Concepts (5 minutes)
Understand the key concepts:
- **IndexSchema**: Defines the structure of your data
- **SearchIndex**: Manages your Redis index
- **VectorQuery**: Performs similarity searches
- **Embeddings**: Vector representations of your data

### 2. First Application (15 minutes)
Build your first semantic search application:
- Follow the [Getting Started notebook](../01_getting_started.ipynb)
- Learn how to index and query documents
- Understand basic filtering

### 3. Explore Features (30+ minutes)
Dive deeper into specific features:
- [Semantic Caching](../03_llmcache.ipynb): Cache LLM responses
- [Message History](../07_message_history.ipynb): Build chatbots
- [Query Routing](../08_semantic_router.ipynb): Route queries intelligently

## Next Steps

After completing the getting started guide, you can:

- **Explore Tutorials**: Learn specific features through hands-on examples
- **Check How-To Guides**: Find solutions for specific tasks
- **Review Use Cases**: See complete application examples
- **Read the API Reference**: Dive deep into the API

## Need Help?

- **Documentation**: You're reading it! Use the search bar to find specific topics
- **GitHub Issues**: Report bugs or request features
- **Community**: Join our Discord or discussions

## Common Questions

**Q: Do I need a Redis Cloud account?**  
A: No, you can use a local Redis instance or any Redis deployment with the Search module.

**Q: What embedding models can I use?**  
A: RedisVL works with any embedding model. See our [Vectorizers guide](../04_vectorizers.ipynb) for popular options.

**Q: Can I use RedisVL with LangChain or LlamaIndex?**  
A: Yes! RedisVL integrates seamlessly with popular LLM frameworks.

**Q: How do I deploy to production?**  
A: Check our deployment guides and best practices in the How-To section.

