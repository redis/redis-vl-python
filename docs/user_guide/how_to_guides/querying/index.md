# Querying How-To Guides

Learn how to query your vector data effectively with different query types and filtering strategies.

## Available Guides

### [Complex Filtering](../../02_complex_filtering.ipynb)
**Level**: Intermediate | **Time**: 20 minutes

Combine multiple filter types (tag, numeric, geo, text) to create sophisticated queries.

**When to use**: You need to filter results by multiple criteria beyond just vector similarity.

**What you'll learn:**
- Combining tag, numeric, and geographic filters
- Using text search with vector search
- Building complex filter expressions
- Performance considerations for complex filters

**Example use cases:**
- "Find similar products under $50 in the Electronics category"
- "Search documents from the last 30 days by a specific author"
- "Find nearby restaurants with high ratings"

---

### [Advanced Queries](../../11_advanced_queries.ipynb)
**Level**: Advanced | **Time**: 30 minutes

Use specialized query types including hybrid search, multi-vector queries, and more.

**When to use**: You need advanced query capabilities beyond basic vector similarity.

**What you'll learn:**
- TextQuery: Full-text search with BM25 scoring
- HybridQuery: Combine text and vector search (Redis 8.4.0+)
- MultiVectorQuery: Search across multiple vector fields
- CountQuery: Count matching records
- RangeQuery: Find vectors within a distance range

**Example use cases:**
- Combining keyword and semantic search
- Searching products by multiple image embeddings
- Finding all documents within a similarity threshold

---

### [SQL to Redis Query Translation](../../12_sql_to_redis_queries.ipynb)
**Level**: Intermediate | **Time**: 15 minutes

Translate familiar SQL queries to Redis query syntax.

**When to use**: You're familiar with SQL and want to understand Redis query equivalents.

**What you'll learn:**
- Mapping SQL WHERE clauses to Redis filters
- Translating JOIN operations
- Converting aggregations
- Understanding Redis query limitations

**Example use cases:**
- Migrating from a SQL-based search system
- Understanding Redis query capabilities
- Building query builders

---

## Query Type Comparison

| Query Type | Use Case | Complexity | Redis Version |
|------------|----------|------------|---------------|
| VectorQuery | Semantic similarity | Basic | All |
| FilterQuery | Metadata filtering | Basic | All |
| TextQuery | Full-text search | Intermediate | All |
| RangeQuery | Distance-based | Intermediate | All |
| HybridQuery | Text + Vector | Advanced | 8.4.0+ |
| MultiVectorQuery | Multiple vectors | Advanced | All |

## Best Practices

1. **Start simple**: Use basic VectorQuery before adding complexity
2. **Filter early**: Apply metadata filters before vector search when possible
3. **Test performance**: Measure query latency with your data
4. **Use appropriate indexes**: Ensure your schema supports your query types
5. **Cache results**: Consider caching for frequently-run queries

## Related Resources

- [Getting Started](../../getting_started/index.md): Learn basic querying
- [Optimization Guides](../optimization/index.md): Improve query performance
- [API Reference](https://docs.redisvl.com/en/stable/api/index.html): Detailed API documentation

## Troubleshooting

**Query returns no results**
- Check your filter syntax
- Verify data exists in the index
- Ensure vector dimensions match

**Query is slow**
- Review your index configuration
- Consider adding metadata indexes
- Check filter selectivity

**Syntax errors**
- Refer to the Redis query syntax documentation
- Use the query builder helpers in RedisVL
- Check for typos in field names

