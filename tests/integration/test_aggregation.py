import pytest
from redis.commands.search.aggregation import AggregateResult
from redis.commands.search.result import Result

from redisvl.redis.connection import compare_versions

from redisvl.index import SearchIndex
from redisvl.query import HybridAggregationQuery
from redisvl.query.filter import (
    FilterExpression,
    Geo,
    GeoRadius,
    Num,
    Tag,
    Text,
)
from redisvl.redis.utils import array_to_buffer


@pytest.fixture
def index(sample_data, redis_url):
    index = SearchIndex.from_dict(
        {
            "index": {
                "name": "user_index",
                "prefix": "v1",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "credit_score", "type": "tag"},
                {"name": "job", "type": "text"},
                {"name": "description", "type": "text"},
                {"name": "age", "type": "numeric"},
                {"name": "last_updated", "type": "numeric"},
                {"name": "location", "type": "geo"},
                {
                    "name": "user_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 3,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
            ],
        },
        redis_url=redis_url,
    )

    # create the index (no data yet)
    index.create(overwrite=True)

    # prepare and load the data
    def hash_preprocess(item: dict) -> dict:
        return {
            **item,
            "user_embedding": array_to_buffer(item["user_embedding"], "float32"),
        }

    index.load(sample_data, preprocess=hash_preprocess)

    # run the test
    yield index

    # clean up
    index.delete(drop=True)


def test_aggregation_query(index):
    redis_version = index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]

    hybrid_query = HybridAggregationQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        return_fields=return_fields,
    )

    results = index.aggregate_query(hybrid_query)
    assert isinstance(results, list)
    assert len(results) == 7
    for doc in results:
        assert doc["user"] in [
            "john",
            "derrick",
            "nancy",
            "tyler",
            "tim",
            "taimur",
            "joe",
            "mary",
        ]
        assert int(doc["age"]) in [18, 14, 94, 100, 12, 15, 35]
        assert doc["job"] in ["engineer", "doctor", "dermatologist", "CEO", "dentist"]
        assert doc["credit_score"] in ["high", "low", "medium"]

    hybrid_query = HybridAggregationQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        num_results=3,
    )

    results = index.aggregate_query(hybrid_query)
    assert len(results) == 3
    assert (
        results[0]["hybrid_score"]
        >= results[1]["hybrid_score"]
        >= results[2]["hybrid_score"]
    )


def test_empty_query_string():
    text = ""
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]

    # test if text is empty
    with pytest.raises(ValueError):
        hybrid_query = HybridAggregationQuery(
            text=text, text_field_name=text_field, vector=vector, vector_field_name=vector_field
        )

    # test if text becomes empty after stopwords are removed
    text = "with a for but and"  # will all be removed as default stopwords
    with pytest.raises(ValueError):
        hybrid_query = HybridAggregationQuery(
            text=text, text_field_name=text_field, vector=vector, vector_field_name=vector_field
        )

def test_aggregation_query_filter(index):
    redis_version = index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]
    filter_expression = (Tag("credit_score") == ("high")) & (Num("age") > 30)

    hybrid_query = HybridAggregationQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        filter_expression=filter_expression,
        return_fields=return_fields,
    )

    results = index.aggregate_query(hybrid_query)
    assert len(results) == 3
    for result in results:
        assert result["credit_score"] == "high"
        assert int(result["age"]) > 30


def test_aggregation_query_with_geo_filter(index):
    redis_version = index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]
    filter_expression = Geo("location") == GeoRadius(37.7749, -122.4194, 1000)

    hybrid_query = HybridAggregationQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        filter_expression=filter_expression,
        return_fields=return_fields,
    )

    results = index.aggregate_query(hybrid_query)
    assert len(results) == 3
    for result in results:
        assert result["location"] is not None


def test_aggregate_query_stopwords(index):
    redis_version = index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"

    hybrid_query = HybridAggregationQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        alpha=0.5,
        stopwords=["medical", "expertise"],
    )

    results = index.aggregate_query(hybrid_query)
    assert len(results) == 7
    for r in results:
        assert r["text_score"] == 0
        assert r["hybrid_score"] == 0.5 * r["vector_similarity"]


def test_aggregate_query_text_filter(index):
    redis_version = index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    filter_expression = (Text("description") == ("medical")) | (Text("job") % ("doct*"))

    hybrid_query = HybridAggregationQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        alpha=0.5,
        filter_expression=filter_expression
        )

    results = index.aggregate_query(hybrid_query)
    assert len(results) == 7
    for result in results:
        assert result["text_score"] == 0
        assert result["hybrid_score"] == 0.5 * result["vector_similarity"]