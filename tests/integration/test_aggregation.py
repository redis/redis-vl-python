from datetime import timedelta

import pytest
from redis.commands.search.aggregation import AggregateResult
from redis.commands.search.result import Result

from redisvl.index import SearchIndex
from redisvl.query import HybridAggregationQuery
from redisvl.query.filter import (
    FilterExpression,
    Geo,
    GeoRadius,
    Num,
    Tag,
    Text,
    Timestamp,
)
from redisvl.redis.utils import array_to_buffer

# TODO expand to multiple schema types and sync + async

vector = ([0.1, 0.1, 0.5],)
vector_field_name = ("user_embedding",)
return_fields = (
    [
        "user",
        "credit_score",
        "age",
        "job",
        "location",
        "last_updated",
    ],
)
filter_expression = (Tag("credit_score") == "high",)
distance_threshold = (0.2,)


@pytest.fixture
def index(sample_data, redis_url):
    # construct a search index from the schema
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

    # Prepare and load the data
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
    # *=>[KNN 7 @user_embedding $vector AS vector_distance]
    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]

    hybrid_query = HybridAggregationQuery(
        text=text,
        text_field=text_field,
        vector=vector,
        vector_field=vector_field,
        return_fields=return_fields,
    )

    results = index.aggregate_query(hybrid_query)
    assert isinstance(results, list)
    assert len(results) == 7
    for doc in results:
        # ensure all return fields present
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

    # test num_results
    hybrid_query = HybridAggregationQuery(
        text=text,
        text_field=text_field,
        vector=vector,
        vector_field=vector_field,
        num_results=3,
    )

    results = index.aggregate_query(hybrid_query)
    assert len(results) == 3
    assert (
        results[0]["hybrid_score"]
        >= results[1]["hybrid_score"]
        >= results[2]["hybrid_score"]
    )


def test_empty_query_string(index):
    text = ""
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]

    # test if text is empty
    with pytest.raises(ValueError):
        hybrid_query = HybridAggregationQuery(
            text=text, text_field=text_field, vector=vector, vector_field=vector_field
        )

    # test if text becomes empty after stopwords are removed
    text = "with a for but and"  # will all be removed as default stopwords
    with pytest.raises(ValueError):
        hybrid_query = HybridAggregationQuery(
            text=text, text_field=text_field, vector=vector, vector_field=vector_field
        )


def test_aggregate_query_stopwords(index):
    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]
    return
    # test num_results
    hybrid_query = HybridAggregationQuery(
        text=text,
        text_field=text_field,
        vector=vector,
        vector_field=vector_field,
        alpha=0.5,
        stopwords=["medical", "expertise"],
    )

    results = index.aggregate_query(hybrid_query)
    assert len(results) == 7
    for r in results:
        assert r["text_score"] == 0
        assert r["hybrid_score"] == 0.5 * r["vector_similarity"]
