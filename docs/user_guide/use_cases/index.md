# Use Cases

Complete, end-to-end examples showing how to build **production-ready applications** with RedisVL.

## What are Use Cases?

Use case guides show you how to build complete applications by combining multiple RedisVL features. Unlike tutorials (which teach one concept) or how-to guides (which solve one problem), use cases demonstrate full application architectures.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🔍 Semantic Search
:class-card: sd-bg-light

Build a semantic search engine that understands meaning, not just keywords.

**Key Features:**
- Document ingestion and chunking
- Embedding generation
- Vector indexing with filtering
- Result ranking and reranking

**Stack**: RedisVL + OpenAI + FastAPI

+++
📚 See [Getting Started](../01_getting_started.ipynb) + [Hybrid Queries](../02_hybrid_queries.ipynb)
:::

:::{grid-item-card} 💬 Chatbots with Memory

Build a conversational AI with persistent memory across sessions.

**Key Features:**
- Conversation history management
- Context-aware responses
- Session management
- Memory optimization

**Stack**: RedisVL + LangChain + OpenAI

+++
📚 Based on [Message History Tutorial](../07_message_history.ipynb)
:::

:::{grid-item-card} 📄 RAG / Document Q&A

Build a Retrieval-Augmented Generation system for document Q&A.

**Key Features:**
- PDF/document parsing
- Smart chunking strategies
- Hybrid search (keyword + semantic)
- Answer generation with citations

**Stack**: RedisVL + LlamaIndex + LlamaParse

+++
📚 See [Semantic Caching](../03_llmcache.ipynb) + [Rerankers](../06_rerankers.ipynb)
:::

:::{grid-item-card} 🎯 Query Routing System

Route user queries to specialized handlers based on intent.

**Key Features:**
- Intent classification
- Multi-handler architecture
- Fallback strategies
- Performance monitoring

**Stack**: RedisVL + FastAPI

+++
📚 Based on [Semantic Router Tutorial](../08_semantic_router.ipynb)
:::

::::

## Choosing a Use Case

| I want to build... | Start Here | Time Estimate |
|-------------------|------------|---------------|
| A search engine that understands meaning | Semantic Search | 2-3 hours |
| A chatbot with memory | Chatbots with Memory | 1-2 hours |
| A document Q&A system | RAG / Document Q&A | 3-4 hours |
| An intent classifier / router | Query Routing System | 1-2 hours |

## Use Case Structure

Each complete use case includes:

```
┌────────────────────────────────────────────────────────────┐
│  1. OVERVIEW          What you'll build and why            │
├────────────────────────────────────────────────────────────┤
│  2. ARCHITECTURE      System design and components         │
├────────────────────────────────────────────────────────────┤
│  3. PREREQUISITES     Required knowledge and tools         │
├────────────────────────────────────────────────────────────┤
│  4. IMPLEMENTATION    Step-by-step code with explanations  │
├────────────────────────────────────────────────────────────┤
│  5. DEPLOYMENT        How to run in production             │
├────────────────────────────────────────────────────────────┤
│  6. EXTENSIONS        Ideas for customization              │
└────────────────────────────────────────────────────────────┘
```

## Building Your Own Use Case

To combine RedisVL features for your own use case:

1. **Start with Getting Started** - Understand core concepts
2. **Complete relevant tutorials** - Learn the features you need
3. **Reference how-to guides** - Solve specific implementation problems
4. **Combine and customize** - Build your application

## Contributing Use Cases

Have a great use case to share? We'd love to include it!

1. Fork the [repository](https://github.com/redis/redis-vl-python)
2. Create your use case guide following our structure
3. Submit a pull request
4. We'll review and provide feedback

## Related Resources

::::{grid} 3
:gutter: 2

:::{grid-item-card} 📚 Tutorials
Learn individual features step-by-step

[Go to Tutorials](../tutorials/index)
:::

:::{grid-item-card} 🛠️ How-To Guides
Solve specific problems quickly

[Go to How-To Guides](../how_to_guides/index)
:::

:::{grid-item-card} 📖 API Reference
Detailed API documentation

[View API Docs](https://redisvl.com)
:::

::::

