---
myst:
  html_meta:
    "description lang=en": |
      RedisVL extensions - semantic caching, embeddings caching, message history, and routing.
---

# Extensions

Extensions are opinionated, higher-level abstractions built on RedisVL's core primitives. Each extension encapsulates a common AI application pattern, managing its own Redis index internally and exposing a clean, purpose-specific API.

You don't need to understand schemas, indexes, or queries to use extensions—they handle that complexity for you.

## Semantic Cache

LLM API calls are expensive and slow. If users ask similar questions, you're paying to generate similar answers repeatedly. Semantic caching solves this by storing responses and returning cached answers when similar prompts are seen again.

### How It Works

When a prompt arrives, the cache embeds it and searches for similar cached prompts. If a match is found within the configured distance threshold, the cached response is returned immediately—no LLM call needed. If no match is found, you call the LLM, store the prompt-response pair, and return the response.

The key insight is "similar" rather than "identical." Traditional caching requires exact matches. Semantic caching matches by meaning, so "What's the capital of France?" and "Tell me France's capital city" can hit the same cache entry.

### Threshold Tuning

The distance threshold controls how similar prompts must be to match. A strict threshold (low value, like 0.05) requires near-identical prompts. A loose threshold (higher value, like 0.3) matches more liberally.

Too strict, and you miss valid cache hits. Too loose, and you return wrong answers for different questions. Start strict, monitor cache quality in production, and loosen gradually based on observed behavior.

### Multi-Tenant Isolation

In applications serving multiple users or contexts, you often want separate cache spaces. Filters let you scope cache lookups—for example, caching per-user or per-conversation so one user's cached answers don't leak to another.

**Learn more:** {doc}`/user_guide/03_llmcache` covers semantic caching in detail.

## Embeddings Cache

Embedding APIs have per-token costs, and computing the same embedding repeatedly wastes money. The embeddings cache stores computed embeddings and returns them on subsequent requests for the same content.

### How It Works

Unlike semantic cache (which uses similarity search), embeddings cache uses exact key matching. A deterministic hash is computed from the input text and model name. If that hash exists in the cache, the stored embedding is returned. If not, the embedding is computed, stored, and returned.

This is useful when the same content is embedded multiple times—common in applications where users submit similar queries, or where documents are re-processed periodically.

### Wrapping Vectorizers

The embeddings cache can wrap any {doc}`vectorizer </concepts/utilities>`, adding transparent caching. Calling the wrapped vectorizer checks the cache first. This requires no changes to your embedding code—just wrap the vectorizer and caching happens automatically.

## Message History

LLMs are stateless. To have a conversation, you must include previous messages in each prompt. Message history manages this context, storing conversation turns and retrieving them when building prompts.

### Storage Model

Each message includes a role (user, assistant, system, or tool), the message content, a timestamp, and a session identifier. The session tag groups messages into conversations—you might have one session per user, per chat thread, or per agent instance.

### Retrieval Strategies

The simplest retrieval is by recency: get the N most recent messages. This works for short conversations but breaks down when context exceeds the LLM's token limit or when relevant information appeared earlier in a long conversation.

Semantic message history adds vector search. Messages are embedded, and you can retrieve by relevance rather than recency. This is powerful for long conversations where the user might reference something said much earlier, or for agents that need to recall specific instructions from their setup.

### Session Isolation

Session tags are critical for multi-user applications. Each user's conversation should be isolated, so retrieving context for User A doesn't include messages from User B. The session tag provides this isolation, and you can structure sessions however makes sense—per-user, per-thread, per-agent, or any other grouping.

**Learn more:** {doc}`/user_guide/07_message_history` explains conversation management in detail.

## Semantic Router

Semantic routing classifies queries into predefined categories based on meaning. It's a lightweight alternative to classification models, useful for intent detection, topic routing, and guardrails.

### How It Works

You define routes, each with a name and a set of reference phrases that represent that category. The router embeds all references and indexes them. At runtime, an incoming query is embedded and compared against all route references. The route whose references are closest to the query wins—if it's within the configured distance threshold.

For example, a customer support router might have routes for "billing," "technical support," and "account management," each with 5-10 reference phrases. When a user asks "I can't log into my account," the router matches it to the "account management" route based on semantic similarity to that route's references.

### Threshold and Aggregation

Each route has its own distance threshold, controlling how close queries must be to match. Routes can also specify how to aggregate distances when multiple references match—taking the average or minimum distance.

If no route matches (all distances exceed their thresholds), the router returns no match. This lets you handle out-of-scope queries gracefully rather than forcing a classification.

### Use Cases

Semantic routing is useful for intent classification (determining what a user wants), topic detection (categorizing content), guardrails (detecting and blocking certain query types), and agent dispatch (sending queries to specialized sub-agents).

**Learn more:** {doc}`/user_guide/08_semantic_router` walks through routing setup in detail.
