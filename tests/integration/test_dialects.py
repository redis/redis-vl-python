import json

import pytest
from redis import Redis
from redis.commands.search.query import Query

from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema.schema import IndexSchema


@pytest.fixture
def sample_data():
    return [
        {
            "name": "Noise-cancelling Bluetooth headphones",
            "description": "Wireless Bluetooth headphones with noise-cancelling technology",
            "connection": {"wireless": True, "type": "Bluetooth"},
            "price": 99.98,
            "stock": 25,
            "colors": ["black", "silver"],
            "embedding": [0.87, -0.15, 0.55, 0.03],
            "embeddings": [[0.56, -0.34, 0.69, 0.02], [0.94, -0.23, 0.45, 0.19]],
        },
        {
            "name": "Wireless earbuds",
            "description": "Wireless Bluetooth in-ear headphones",
            "connection": {"wireless": True, "type": "Bluetooth"},
            "price": 64.99,
            "stock": 17,
            "colors": ["red", "black", "white"],
            "embedding": [-0.7, -0.51, 0.88, 0.14],
            "embeddings": [[0.54, -0.14, 0.79, 0.92], [0.94, -0.93, 0.45, 0.16]],
        },
    ]


@pytest.fixture
def schema_dict():
    return {
        "index": {"name": "products", "prefix": "product", "storage_type": "json"},
        "fields": [
            {"name": "name", "type": "text"},
            {"name": "description", "type": "text"},
            {"name": "connection_type", "path": "$.connection.type", "type": "tag"},
            {"name": "price", "type": "numeric"},
            {"name": "stock", "type": "numeric"},
            {"name": "color", "path": "$.colors.*", "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {"dims": 4, "algorithm": "flat", "distance_metric": "cosine"},
            },
            {
                "name": "embeddings",
                "path": "$.embeddings[*]",
                "type": "vector",
                "attrs": {"dims": 4, "algorithm": "hnsw", "distance_metric": "l2"},
            },
        ],
    }


@pytest.fixture
def index(sample_data, redis_url, schema_dict):
    index_schema = IndexSchema.from_dict(schema_dict)
    redis_client = Redis.from_url(redis_url)
    index = SearchIndex(index_schema, redis_client)
    index.create(overwrite=True, drop=True)
    index.load(sample_data)
    yield index
    index.delete(drop=True)


def test_dialect_3_json(index, sample_data):
    # Create a VectorQuery with dialect 3
    vector_query = VectorQuery(
        vector=[0.23, 0.12, -0.03, 0.98],
        vector_field_name="embedding",
        return_fields=["name", "description", "price"],
        dialect=3,
    )

    # Execute the query
    results = index.query(vector_query)

    # Print the results
    print("VectorQuery Results:")
    print(results)

    # Assert the expected format of the results
    assert len(results) > 0
    for result in results:
        assert not isinstance(result["name"], list)
        assert not isinstance(result["description"], list)
        assert not isinstance(result["price"], (list, str))

    # Create a FilterQuery with dialect 3
    filter_query = FilterQuery(
        filter_expression=Tag("color") == "black",
        return_fields=["name", "description", "price"],
        dialect=3,
    )

    # Execute the query
    results = index.query(filter_query)

    # Print the results
    print("FilterQuery Results:")
    print(results)

    # Assert the expected format of the results
    assert len(results) > 0
    for result in results:
        assert not isinstance(result["name"], list)
        assert not isinstance(result["description"], list)
        assert not isinstance(result["price"], (list, str))
