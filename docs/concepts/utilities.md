---
myst:
  html_meta:
    "description lang=en": |
      RedisVL utilities - vectorizers for embeddings and rerankers for result optimization.
---

# Utilities

Utilities are optional components that enhance search workflows. They're not required—you can bring your own embeddings and skip reranking—but they simplify common tasks.

## Vectorizers

A vectorizer converts text into an embedding vector. Embeddings are dense numerical representations that capture semantic meaning: similar texts produce similar vectors, enabling similarity search.

### Why Vectorizers Matter

Creating embeddings requires calling an embedding model—either a cloud API (OpenAI, Cohere, etc.) or a local model (HuggingFace sentence-transformers). Each provider has different APIs, authentication methods, and response formats.

RedisVL vectorizers provide a unified interface across providers. You choose a provider, configure it once, and use the same methods regardless of which service is behind it. This makes it easy to switch providers, compare models, or use different providers in different environments.

### The Dimensionality Contract

Every embedding model produces vectors of a specific size (dimensionality). OpenAI's text-embedding-3-small produces 1536-dimensional vectors. Other models produce 384, 768, 1024, or other sizes.

Your schema's vector field must specify the same dimensionality as your embedding model. If there's a mismatch—your model produces 1536-dimensional vectors but your schema expects 768—you'll get errors when loading data or running queries.

This constraint means you should choose your embedding model before designing your schema, and changing models requires rebuilding your index.

### Batching and Performance

Embedding APIs have rate limits and per-request overhead. Embedding one text at a time is inefficient. Vectorizers support batch embedding, sending multiple texts in a single request. This dramatically improves throughput for indexing large datasets.

Vectorizers handle batching internally, breaking large batches into provider-appropriate chunks and respecting rate limits. You provide a list of texts; the vectorizer manages the logistics.

### Supported Providers

RedisVL includes vectorizers for OpenAI, Azure OpenAI, Cohere, HuggingFace (local), Mistral, Google Vertex AI, AWS Bedrock, VoyageAI, and others. See the {doc}`/api/vectorizer` for the complete list. You can also create custom vectorizers that wrap any embedding function.

## Rerankers

A reranker takes initial search results and reorders them by relevance to the query. It's a second-stage filter that improves precision after the first-stage retrieval.

### Why Reranking Works

Vector search uses bi-encoder models: the query and documents are embedded independently, then compared by vector distance. This is fast but approximate—the embedding captures general meaning, not the specific relationship between query and document.

Rerankers use cross-encoder models that score query-document pairs directly. The model sees both the query and document together and predicts a relevance score. This is more accurate but slower, because each candidate requires a separate model inference.

The combination is powerful: use fast vector search to retrieve a broad set of candidates (high recall), then use the slower but more accurate reranker to select the best results (high precision).

### The Recall-Precision Trade-off

With only vector search, you might retrieve 10 results and hope the best one is in there. With reranking, you can retrieve 50 candidates—casting a wider net—then rerank to find the 5 best. The initial retrieval prioritizes recall (not missing relevant documents); reranking prioritizes precision (surfacing the most relevant ones).

### Cost and Latency

Reranking adds latency (typically 50-200ms depending on the provider and number of candidates) and cost (API-based rerankers charge per request). These trade-offs are usually worthwhile when result quality matters, but you should measure the impact for your use case.

### Supported Providers

RedisVL includes rerankers for HuggingFace cross-encoders (local), Cohere Rerank API, and VoyageAI Rerank API.

## Two-Stage Retrieval

The most effective retrieval pipelines combine both utilities: vectorize the query, retrieve a candidate set with vector search, then rerank to select the final results.

This pattern separates recall (finding everything potentially relevant) from precision (selecting the best matches). Vector search handles recall efficiently; reranking handles precision accurately. Together, they deliver better results than either approach alone.

---

**Related concepts:** {doc}`queries` explains how to use embeddings in vector search queries. {doc}`search-and-indexing` covers schema configuration for vector fields.

**Learn more:** {doc}`/user_guide/04_vectorizers` covers embedding providers. {doc}`/user_guide/06_rerankers` explains reranking in practice.

