from redis.commands.search.field import (
    TagField,
    VectorField
)

# Build Schema
def get_schema(size: int):
    return [
        # Tag fields
        TagField("categories", separator = "|"),
        TagField("year", separator = "|"),
        # Vector field (FLAT index with COSINE similarity)
        VectorField(
            "vector",
            "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": 768,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": size,
                "BLOCK_SIZE": size
            }
        )
    ]