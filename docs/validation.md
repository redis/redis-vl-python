# RedisVL Validation System

The RedisVL validation system ensures that data written to Redis indexes conforms to the defined schema. It uses dynamic Pydantic model generation to validate objects before they are stored.

## Key Features

- **Schema-Based Validation**: Validates objects against your index schema definition
- **Dynamic Model Generation**: Creates Pydantic models on the fly based on your schema
- **Type Checking**: Ensures fields contain appropriate data types
- **Field-Specific Validation**:
  - Text and Tag fields must be strings
  - Numeric fields must be integers or floats
  - Geo fields must be properly formatted latitude/longitude strings
  - Vector fields must have the correct dimensions and data types
- **JSON Path Support**: Validates fields extracted from nested JSON structures
- **Fail-Fast Approach**: Stops processing at the first validation error
- **Performance Optimized**: Caches models for repeated validation

## Usage

### Basic Validation

```python
from redisvl.schema.validation import validate_object

# Assuming you have a schema defined
validated_data = validate_object(schema, data)
```

### Storage Integration

The validation is automatically integrated with the storage classes:

```python
from redisvl.index.storage import BaseStorage

# Create storage with schema
storage = BaseStorage(schema=schema, client=redis_client)

# Write data - validation happens automatically
storage.write_one(data)

# Or validate explicitly
validated = storage.validate_object(data)
```

## Field Type Validation

The validation system supports all Redis field types:

### Text Fields

Text fields are validated to ensure they contain string values:

```python
# Valid
{"title": "Hello World"}

# Invalid
{"title": 123}  # Not a string
```

### Tag Fields

Tag fields are validated to ensure they contain string values:

```python
# Valid
{"category": "electronics"}

# Invalid
{"category": 123}  # Not a string
```

### Numeric Fields

Numeric fields must contain integers or floats:

```python
# Valid
{"price": 19.99}
{"quantity": 5}

# Invalid
{"price": "19.99"}  # String, not a number
```

### Geo Fields

Geo fields must contain properly formatted latitude/longitude strings:

```python
# Valid
{"location": "37.7749,-122.4194"}  # San Francisco
{"location": "40.7128,-74.0060"}   # New York

# Invalid
{"location": "invalid"}            # Not in lat,lon format
{"location": "91.0,0.0"}           # Latitude out of range (-90 to 90)
{"location": "0.0,181.0"}          # Longitude out of range (-180 to 180)
```

### Vector Fields

Vector fields must contain arrays with the correct dimensions and data types:

```python
# Valid
{"embedding": [0.1, 0.2, 0.3, 0.4]}  # 4-dimensional float vector
{"embedding": b'\x00\x01\x02\x03'}   # Raw bytes (dimensions not checked)

# Invalid
{"embedding": [0.1, 0.2, 0.3]}        # Wrong dimensions
{"embedding": "not a vector"}         # Wrong type
{"embedding": [0.1, "text", 0.3]}     # Mixed types
```

For integer vectors, the values must be within the appropriate range:

- **INT8**: -128 to 127
- **INT16**: -32,768 to 32,767

```python
# Valid INT8 vector
{"int_vector": [1, 2, 3]}

# Invalid INT8 vector
{"int_vector": [1000, 2000, 3000]}  # Values out of range
```

## Nested JSON Validation

The validation system supports extracting and validating fields from nested JSON structures:

```python
# Schema with JSON paths
fields = {
    "id": Field(name="id", type=FieldTypes.TAG),
    "title": Field(name="title", type=FieldTypes.TEXT, path="$.content.title"),
    "rating": Field(name="rating", type=FieldTypes.NUMERIC, path="$.metadata.rating")
}

# Nested JSON data
data = {
    "id": "doc1",
    "content": {
        "title": "Hello World"
    },
    "metadata": {
        "rating": 4.5
    }
}

# Validation extracts fields using JSON paths
validated = validate_object(schema, data)
# Result: {"id": "doc1", "title": "Hello World", "rating": 4.5}
```

## Error Handling

The validation system uses a fail-fast approach, raising a `ValueError` when validation fails:

```python
try:
    validated = validate_object(schema, data)
except ValueError as e:
    print(f"Validation error: {e}")
    # Handle the error
```

The error message includes information about the field that failed validation.

## Optional Fields

All fields are considered optional during validation. If a field is missing, it will be excluded from the validated result:

```python
# Schema with multiple fields
fields = {
    "id": Field(name="id", type=FieldTypes.TAG),
    "title": Field(name="title", type=FieldTypes.TEXT),
    "rating": Field(name="rating", type=FieldTypes.NUMERIC)
}

# Data with missing fields
data = {
    "id": "doc1",
    "title": "Hello World"
    # rating is missing
}

# Validation succeeds with partial data
validated = validate_object(schema, data)
# Result: {"id": "doc1", "title": "Hello World"}
```

## Performance Considerations

The validation system is optimized for performance:

- **Model Caching**: Pydantic models are cached by schema name to avoid regeneration
- **Lazy Validation**: Fields are validated only when needed
- **Fail-Fast Approach**: Processing stops at the first validation error

For large datasets, validation can be a significant part of the processing time. If you need to write many objects with the same structure, consider validating a sample first to ensure correctness.

## Limitations

- **JSON Path**: The current implementation only supports simple dot notation paths (e.g., `$.field.subfield`). Array indexing is not supported.
- **Vector Bytes**: When vectors are provided as bytes, the dimensions cannot be validated.
- **Custom Validators**: The current implementation does not support custom user-defined validators.

## Best Practices

1. **Define Clear Schemas**: Be explicit about field types and constraints
2. **Pre-validate Critical Data**: For large datasets, validate a sample before processing everything
3. **Handle Validation Errors**: Implement proper error handling for validation failures
4. **Use JSON Paths Carefully**: Test nested JSON extraction to ensure paths are correctly defined
5. **Consider Optional Fields**: Decide which fields are truly required for your application

## Integration with Storage Classes

The validation system is fully integrated with the storage classes:

- **BaseStorage**: For hash-based storage, validates each field individually
- **JsonStorage**: For JSON storage, extracts and validates fields from nested structures

Each storage class automatically validates data before writing to Redis, ensuring data integrity. 