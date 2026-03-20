---
name: redisvl-mcp-production-example
description: Companion production-oriented example for the RedisVL MCP server specification.
metadata:
  status: draft
  audience: RedisVL maintainers and reviewers
  objective: Provide a concrete, production-evaluable usage narrative without bloating the normative MCP specification.
---

# RedisVL MCP Production Example

This document is a companion to [MCP.md](./MCP.md). It is intentionally narrative and example-driven. The normative server contract lives in the main spec.

## Why This Example Exists

The MCP specification is easier to evaluate when grounded in a realistic deployment. This example uses a Redis Enterprise customer because that is a strong production reference point, but the same RedisVL MCP design is intended to work with Redis Cloud and open-source Redis Stack instances, including local Docker deployments, provided the required index and Search capabilities already exist.

## User Story

As a platform team at a company running Redis Enterprise for internal knowledge retrieval, we want to expose our existing Redis vector indexes through MCP so internal AI assistants can perform low-latency, metadata-filtered search over approved enterprise content without copying data into another vector store or hand-recreating index schemas.

## Scenario

An enterprise platform team already operates a Redis-backed knowledge index called `internal_knowledge`. The index contains:

- operational runbooks
- support knowledge base articles
- release notes
- incident summaries

The team has already standardized on Redis as the serving layer for retrieval. Multiple internal assistants need access to the same retrieval surface:

- an engineering support copilot in Slack
- a developer portal assistant
- an incident review assistant

The platform team does not want each assistant team to:

- reimplement Redis query logic
- duplicate the index into a separate vector database
- manually re-describe the Redis schema in every client integration

Instead, they publish one RedisVL MCP server configuration with one approved index binding in v1. The MCP server attaches to an existing index, inspects its schema at startup, and exposes a stable tool contract to AI clients.

This is intentionally simplified for v1 review. In a larger deployment, the same content domains could reasonably be split across multiple Redis indexes, such as separate bindings for runbooks, support KB content, release notes, or incident history. That would create a future need for one MCP server to route across multiple configured index bindings while keeping a coherent tool surface for clients.

## Why MCP Helps

MCP gives the platform team a standard tool boundary:

- AI clients can use the same `search-records` contract.
- The Redis index stays the source of truth for field definitions and search behavior.
- The vectorizer remains explicit and reviewable, which matters when embedding model choice is governed separately from index operations.
- Metadata filters remain available to enforce application-level narrowing such as team, region, product, and severity.
- The MCP surface can stay read-only for assistant clients, which avoids exposing direct write access to the internal knowledge index.

## Deployment Sketch

1. Redis already hosts the `internal_knowledge` index.
2. The platform team provisions a small stdio MCP process near the client runtime.
3. The MCP server connects to Redis using a normal Redis URL.
4. At startup, the server inspects `internal_knowledge` and reconstructs the schema.
5. The server applies any small override needed for incomplete vector metadata.
6. The configured vectorizer embeds user queries for vector or hybrid search.
7. Internal assistants call the MCP tool instead of talking to Redis directly.

This pattern works across:

- Redis Enterprise in a self-managed production environment
- Redis Cloud instances used by product teams
- open-source Redis Stack, including Docker-based local and CI environments

The behavioral contract stays the same. The operational controls around networking, auth, and tenancy vary by deployment.

## Example MCP Config

```yaml
server:
  redis_url: ${REDIS_URL}
  read_only: true

indexes:
  knowledge:
    redis_name: internal_knowledge

    vectorizer:
      class: OpenAITextVectorizer
      model: text-embedding-3-small
      api_config:
        api_key: ${OPENAI_API_KEY}

    schema_overrides:
      fields:
        - name: embedding
          type: vector
          attrs:
            dims: 1536
            datatype: float32

    runtime:
      text_field_name: content
      vector_field_name: embedding
      default_embed_text_field: content
      default_limit: 8
      max_limit: 25
      skip_embedding_if_present: true
      startup_timeout_seconds: 30
      request_timeout_seconds: 45
      max_concurrency: 16
```

Why this is realistic:

- The index already exists and is discovered automatically.
- The v1 config still targets one bound index, but the surrounding YAML shape can grow to multiple bindings later.
- The vectorizer is still configured manually.
- `schema_overrides` is available if Redis inspection does not fully reconstruct vector attrs.
- Runtime field mappings stay explicit so the MCP server does not guess among multiple text-like fields.
- Assistant clients are intentionally limited to read-only retrieval against the internal knowledge index.

## Example Search Calls

### Vector search for incident guidance

Request:

```json
{
  "query": "How do we mitigate elevated cache miss rate after a regional failover?",
  "search_type": "vector",
  "limit": 5,
  "filter": {
    "and": [
      { "field": "team", "op": "eq", "value": "platform" },
      { "field": "severity", "op": "in", "value": ["sev1", "sev2"] },
      { "field": "region", "op": "eq", "value": "eu-central" }
    ]
  },
  "return_fields": ["title", "content", "source_type", "last_reviewed_at"]
}
```

Why the enterprise customer cares:

- Semantic retrieval finds similar operational incidents even when the exact wording differs.
- Filters keep the result set scoped to the right team, severity band, and region.

### Hybrid search for release-note lookup

Request:

```json
{
  "query": "deprecation of legacy cache invalidation flow",
  "search_type": "hybrid",
  "limit": 3,
  "filter": {
    "field": "product",
    "op": "eq",
    "value": "developer-platform"
  },
  "return_fields": ["title", "content", "release_version"]
}
```

Why the enterprise customer cares:

- Hybrid search combines exact phrase hits with semantic similarity.
- The same MCP request works whether the server uses native Redis hybrid search or the `AggregateHybridQuery` fallback.
- The assistant can ground answers in specific release-note entries instead of relying on model memory.

## Evaluation Checklist For Reviewers

This example should make the value of the MCP design easy to evaluate:

- The customer already has Redis indexes and wants to reuse them.
- The server discovers index structure instead of forcing duplicate schema definition.
- The vectorizer is still explicit, which keeps embedding behavior auditable.
- The same pattern applies across Enterprise, Cloud, and OSS deployments.
- The assistant-facing MCP surface can remain read-only even if the underlying index is maintained by separate ingestion systems.
- The scenario also illustrates why future multi-index support may matter as teams split distinct content domains into separate Redis indexes.
- The MCP layer standardizes how multiple assistants consume the same Redis retrieval system.
