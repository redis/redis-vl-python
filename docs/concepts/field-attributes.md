---
myst:
  html_meta:
    "description lang=en": |
      RedisVL field attributes - configuring sortable, no_index, index_missing, and other field options.
---

# Field Attributes

Field attributes customize how Redis indexes and searches your data. Each field type has specific attributes that control indexing behavior, search capabilities, and storage options.

## Common Attributes

These attributes are available on most non-vector field types (text, tag, numeric, geo).

### sortable

Enables sorting results by this field. Without `sortable`, you cannot use the field in `ORDER BY` clauses.

**Trade-off**: Sortable fields consume additional memory to maintain a sorted index.

```yaml
# YAML
- name: created_at
  type: numeric
  attrs:
    sortable: true
```

```python
# Python
{"name": "created_at", "type": "numeric", "attrs": {"sortable": True}}
```

**Use when**: You need to sort search results by this field (e.g., "newest first", "highest price").

### no_index

Stores the field without indexing it. The field value is available in search results but cannot be used in queries or filters.

**Important**: `no_index` only makes sense when combined with `sortable: true`. A field that is neither indexed nor sortable serves no purpose in the schema.

```yaml
# YAML - Store for sorting but don't index for search
- name: internal_score
  type: numeric
  attrs:
    sortable: true
    no_index: true
```

**Use when**: You want to sort by a field but never filter on it, saving index space.

### index_missing

Allows searching for documents that don't have this field. When enabled, you can use `ISMISSING` queries to find documents where the field is absent or null.

```yaml
# YAML
- name: optional_category
  type: tag
  attrs:
    index_missing: true
```

```python
# Python
{"name": "optional_category", "type": "tag", "attrs": {"index_missing": True}}
```

**Use when**: Your data has optional fields and you need to query for documents missing those fields.

**Query example**:
```python
from redisvl.query.filter import Tag

# Find documents where category is missing
filter_expr = Tag("optional_category").ismissing()
```

## Text Field Attributes

Text fields support full-text search with these additional attributes.

### weight

Controls the importance of this field in relevance scoring. Higher weights make matches in this field rank higher.

```yaml
- name: title
  type: text
  attrs:
    weight: 2.0  # Title matches count double

- name: description
  type: text
  attrs:
    weight: 1.0  # Default weight
```

**Use when**: Some text fields are more important than others for search relevance.

### no_stem

Disables stemming for this field. By default, Redis applies stemming so "running" matches "run". Disable when exact word forms matter.

```yaml
- name: product_code
  type: text
  attrs:
    no_stem: true
```

**Use when**: Field contains codes, identifiers, or technical terms where stemming would cause incorrect matches.

### withsuffixtrie

Maintains a suffix trie for optimized suffix and contains queries. Enables efficient `*suffix` and `*contains*` searches.

```yaml
- name: email
  type: text
  attrs:
    withsuffixtrie: true
```

**Use when**: You need to search for patterns like `*@gmail.com` or `*smith*`.

**Trade-off**: Increases memory usage and index build time.

### phonetic_matcher

Enables phonetic matching using the specified algorithm. Matches words that sound similar.

```yaml
- name: name
  type: text
  attrs:
    phonetic_matcher: "dm:en"  # Double Metaphone, English
```

**Supported values**: `dm:en` (Double Metaphone English), `dm:fr` (French), `dm:pt` (Portuguese), `dm:es` (Spanish)

**Use when**: Searching names or words where spelling variations should match (e.g., "Smith" matches "Smyth").

### index_empty

Allows indexing and searching for empty strings. By default, empty strings are not indexed.

```yaml
- name: middle_name
  type: text
  attrs:
    index_empty: true
```

**Use when**: Empty string is a meaningful value you need to query for.

### unf (Un-Normalized Form)

Preserves the original value for sortable fields without normalization. By default, sortable text fields are lowercased for consistent sorting.

**Requires**: `sortable: true`

```yaml
- name: title
  type: text
  attrs:
    sortable: true
    unf: true  # Keep original case for sorting
```

**Use when**: You need case-sensitive sorting or must preserve exact original values.

## Tag Field Attributes

Tag fields are for exact-match filtering on categorical data.

### separator

Specifies the character that separates multiple tags in a single field value. Default is comma (`,`).

```yaml
- name: categories
  type: tag
  attrs:
    separator: "|"  # Use pipe instead of comma
```

**Use when**: Your tag values contain commas, or you're using a different delimiter in your data.

### case_sensitive

Makes tag matching case-sensitive. By default, tags are lowercased for matching.

```yaml
- name: product_sku
  type: tag
  attrs:
    case_sensitive: true
```

**Use when**: Tag values are case-sensitive identifiers (SKUs, codes, etc.).

### withsuffixtrie

Same as text fields—enables efficient suffix and contains queries on tags.

```yaml
- name: email_domain
  type: tag
  attrs:
    withsuffixtrie: true
```

### index_empty

Allows indexing empty tag values.

```yaml
- name: optional_tags
  type: tag
  attrs:
    index_empty: true
```

## Numeric Field Attributes

Numeric fields support range queries and sorting.

### unf (Un-Normalized Form)

For sortable numeric fields, preserves the exact numeric representation without normalization.

**Requires**: `sortable: true`

```yaml
- name: price
  type: numeric
  attrs:
    sortable: true
    unf: true
```

**Note**: Numeric fields do not support `index_empty` (empty numeric values are not meaningful).

## Geo Field Attributes

Geo fields store geographic coordinates for location-based queries.

Geo fields support the common attributes (`sortable`, `no_index`, `index_missing`) but have no geo-specific attributes. The field value should be a string in `"longitude,latitude"` format.

```yaml
- name: location
  type: geo
  attrs:
    sortable: true
```

**Note**: Geo fields do not support `index_empty` (empty coordinates are not meaningful).

## Vector Field Attributes

Vector fields have a different attribute structure. See {doc}`/api/schema` for complete vector field documentation.

Key vector attributes:
- `dims`: Vector dimensionality (required)
- `algorithm`: `flat`, `hnsw`, or `svs-vamana`
- `distance_metric`: `COSINE`, `L2`, or `IP`
- `datatype`: `float16`, `float32`, `float64`, or `bfloat16`
- `index_missing`: Allow searching for documents without vectors

```yaml
- name: embedding
  type: vector
  attrs:
    algorithm: hnsw
    dims: 768
    distance_metric: cosine
    datatype: float32
    index_missing: true  # Handle documents without embeddings
```

## Redis-Specific Subtleties

### Modifier Ordering

RediSearch has specific requirements for the order of field modifiers. RedisVL handles this automatically, but it's useful to understand:

**Canonical order**: `INDEXEMPTY` → `INDEXMISSING` → `SORTABLE` → `UNF` → `NOINDEX`

If you're debugging raw Redis commands, ensure modifiers appear in this order.

### Field Type Limitations

Not all attributes work with all field types:

| Attribute | Text | Tag | Numeric | Geo | Vector |
|-----------|------|-----|---------|-----|--------|
| `sortable` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `no_index` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `index_missing` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `index_empty` | ✓ | ✓ | ✗ | ✗ | ✗ |
| `unf` | ✓ | ✗ | ✓ | ✗ | ✗ |
| `withsuffixtrie` | ✓ | ✓ | ✗ | ✗ | ✗ |

### JSON Path for Nested Fields

When using JSON storage, use the `path` attribute to index nested fields:

```yaml
- name: author_name
  type: text
  path: $.metadata.author.name
  attrs:
    sortable: true
```

The `name` becomes the field's alias in queries, while `path` specifies where to find the data.

## Complete Example

```yaml
version: "0.1.0"
index:
  name: products
  prefix: product
  storage_type: json

fields:
  # Full-text searchable with high relevance
  - name: title
    type: text
    path: $.title
    attrs:
      weight: 2.0
      sortable: true

  # Exact-match categories
  - name: category
    type: tag
    path: $.category
    attrs:
      separator: "|"
      index_missing: true

  # Sortable price with range queries
  - name: price
    type: numeric
    path: $.price
    attrs:
      sortable: true

  # Store-only field for sorting
  - name: internal_rank
    type: numeric
    path: $.internal_rank
    attrs:
      sortable: true
      no_index: true

  # Vector embeddings
  - name: embedding
    type: vector
    path: $.embedding
    attrs:
      algorithm: hnsw
      dims: 768
      distance_metric: cosine

  # Location search
  - name: store_location
    type: geo
    path: $.location
```

**Learn more:** {doc}`/api/schema` provides the complete API reference for all field types and attributes.

