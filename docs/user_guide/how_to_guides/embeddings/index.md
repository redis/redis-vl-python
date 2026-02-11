# Embeddings How-To Guides

Learn how to work with vector embeddings effectively in RedisVL.

## Available Guides

### [Choosing Vectorizers](../../04_vectorizers.ipynb)
**Level**: Beginner | **Time**: 25 minutes

Select and configure the right embedding model for your use case.

**What you'll learn:**
- Available embedding providers (OpenAI, Cohere, HuggingFace, etc.)
- Comparing embedding models
- Configuring vectorizers
- Custom embedding integrations

**When to use**: You're setting up a new project or evaluating embedding models.

---

### [Caching Embeddings](../../10_embeddings_cache.ipynb)
**Level**: Intermediate | **Time**: 20 minutes

Improve performance and reduce costs by caching embeddings.

**What you'll learn:**
- Setting up an embedding cache
- Cache hit/miss strategies
- Managing cache size
- Measuring cache effectiveness

**When to use**: You're generating embeddings for the same content repeatedly.

---

## Choosing an Embedding Model

Consider these factors:

| Factor | Questions to Ask |
|--------|------------------|
| **Domain** | Is your data general or domain-specific? |
| **Language** | Do you need multilingual support? |
| **Dimensions** | What's your performance vs. accuracy trade-off? |
| **Cost** | API-based or self-hosted? |
| **Latency** | Real-time or batch processing? |

## Popular Embedding Models

- **OpenAI text-embedding-3-small**: Fast, cost-effective, good for most use cases
- **OpenAI text-embedding-3-large**: Higher accuracy, more expensive
- **Cohere embed-v3**: Excellent for search and retrieval
- **HuggingFace models**: Free, self-hosted options
- **Voyage AI**: Optimized for retrieval tasks

## Best Practices

1. **Test multiple models**: Evaluate on your specific data
2. **Cache aggressively**: Embeddings are expensive to generate
3. **Batch when possible**: Reduce API calls
4. **Monitor costs**: Track embedding generation expenses
5. **Version your models**: Document which model version you're using

## Related Resources

- [Vectorizers Guide](../../04_vectorizers.ipynb)
- [Embeddings Cache Guide](../../10_embeddings_cache.ipynb)
- [Optimization Guides](../optimization/index.md)

