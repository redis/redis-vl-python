# Hybrid Filtering

When developing vector search applications, it's common to incorporate additional filtering capabilities beyond vector similarity operations. This includes filtering specific fields within a record using simple filters. This combined approach of filtering on fields and applying vector similarity operations is known as hybrid filtering or metadata filtering.

## Applying Simple Filters

Hybrid filtering involves leveraging simple filters to narrow down search results based on specific field values. This allows for more targeted and precise retrieval of records that meet certain criteria. By combining these traditional filtering techniques with vector similarity operations, developers can enhance the search functionality and improve the relevance of the results.

## Filtering in Redis for E-commerce Recommendation Systems

Redis provides various filtering options that can be leveraged in e-commerce recommendation systems to retrieve relevant products based on specific criteria. Let's explore three types of filters: Tag filters, Numeric filters, and Text (match) filters.

### Tag Filters

Tag filters are useful when you want to filter documents based on specific tags associated with them. In RediSearch, you can use the tag field type for this purpose. Tags can have multiple words or include punctuation marks, and they are enclosed in curly braces.

Example:

```bash
@category:{ Electronics | Mobile Phones }
```

Multiple tag filters in the same query create a union of documents containing any of the specified tags. To create an intersection of documents containing all tags, you can repeat the tag filter for each tag.

```bash
# Return all documents containing both tags 'Electronics' and 'Mobile Phones'
@category:{ Electronics } @category:{ Mobile Phones }
```

### Numeric Filters

Numeric filters are handy when you want to filter documents based on numerical ranges, such as price or rating. In RediSearch, if a field in the schema is defined as NUMERIC, you can use the filter argument in Redis requests or specify filtering rules in the query.

The syntax for a numeric filter is `@field:[{min} {max}]`, where `min` and `max` represent the range values.
Numeric filters can be inclusive, and you can use special symbols like -inf, inf, and +inf to represent negative and positive infinity. Exclusive min or max values can be expressed using parentheses.

```bash
@price:[100 200]

# Price greater than 100
@price:[(100 inf]

# Price between 100 and 200 (exclusive)
@price:[(100 (200]
```

### Text (Match) Filters

Text filters, also known as match filters, are used to search for documents that match specific keywords or phrases. In RediSearch, you can use the text field type for this purpose. Text filters can be applied to fields like product titles, descriptions, or any other textual data.

```bash
@title:redis @description:"in-memory database"
```

In the above example, the query filters documents based on the presence of the word "redis" in the title field and the phrase "in-memory database" in the description field.
