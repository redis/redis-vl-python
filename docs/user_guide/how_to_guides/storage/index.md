# Storage How-To Guides

Learn how to configure data storage in Redis for optimal performance and flexibility.

## Available Guides

### [Hash vs JSON Storage](../../05_hash_vs_json.ipynb)
**Level**: Beginner | **Time**: 15 minutes

Understand the differences between Hash and JSON storage formats and choose the right one for your use case.

**What you'll learn:**
- Differences between Hash and JSON storage
- Performance characteristics of each
- When to use each format
- Migration between formats

**When to use**: You're setting up a new index or optimizing storage.

---

## Storage Format Comparison

| Feature | Hash | JSON |
|---------|------|------|
| **Structure** | Flat key-value pairs | Nested documents |
| **Query Support** | Basic fields only | JSONPath queries |
| **Performance** | Faster for simple data | Better for complex data |
| **Memory** | More efficient | Slightly higher overhead |
| **Flexibility** | Limited nesting | Full nesting support |
| **Best For** | Simple schemas | Complex documents |

## Decision Guide

**Use Hash when:**
- Your data is flat (no nesting)
- You need maximum performance
- Memory efficiency is critical
- You don't need JSONPath queries

**Use JSON when:**
- Your data has nested structures
- You need flexible schema evolution
- You want to query nested fields
- You're working with document-like data

## Best Practices

1. **Start with your data structure**: Let your data shape drive the decision
2. **Consider query patterns**: What fields will you filter on?
3. **Measure performance**: Test both formats with your data
4. **Plan for growth**: Will your schema evolve?
5. **Document your choice**: Record why you chose a format

## Migration Strategies

### Hash to JSON
```python
# Export data from Hash index
# Transform to JSON format
# Create new JSON index
# Import data
# Update application code
# Switch traffic
```

### JSON to Hash
```python
# Flatten JSON documents
# Create new Hash index
# Import flattened data
# Update application code
# Switch traffic
```

## Related Resources

- [Hash vs JSON Guide](../../05_hash_vs_json.ipynb)
- [Getting Started](../../getting_started/index.md)
- [Optimization Guides](../optimization/index.md)

## Troubleshooting

**Can't query nested fields with Hash**
- Solution: Use JSON storage or flatten your data

**JSON queries are slow**
- Solution: Ensure proper indexing on queried fields
- Consider Hash if you don't need nesting

**High memory usage**
- Solution: Try Hash format for simpler data
- Implement data expiration policies

