import pytest

from redisvl.index import SearchIndex
from redisvl.query import HybridQuery
from redisvl.query.filter import FilterExpression, Geo, GeoRadius, Num, Tag, Text
from redisvl.redis.utils import array_to_buffer
from tests.conftest import skip_if_redis_version_below


@pytest.fixture
def index(sample_data, redis_url, worker_id):
    index = SearchIndex.from_dict(
        {
            "index": {
                "name": f"user_index_{worker_id}",
                "prefix": f"v1_{worker_id}",
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
    skip_if_redis_version_below(index.client, "7.2.0")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]

    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        return_fields=return_fields,
    )

    results = index.query(hybrid_query)
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

    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        num_results=3,
    )

    results = index.query(hybrid_query)
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
        hybrid_query = HybridQuery(
            text=text,
            text_field_name=text_field,
            vector=vector,
            vector_field_name=vector_field,
        )

    # test if text becomes empty after stopwords are removed
    text = "with a for but and"  # will all be removed as default stopwords
    with pytest.raises(ValueError):
        hybrid_query = HybridQuery(
            text=text,
            text_field_name=text_field,
            vector=vector,
            vector_field_name=vector_field,
        )


def test_aggregation_query_with_filter(index):
    skip_if_redis_version_below(index.client, "7.2.0")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]
    filter_expression = (Tag("credit_score") == ("high")) & (Num("age") > 30)

    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        filter_expression=filter_expression,
        return_fields=return_fields,
    )

    results = index.query(hybrid_query)
    assert len(results) == 2
    for result in results:
        assert result["credit_score"] == "high"
        assert int(result["age"]) > 30


def test_aggregation_query_with_geo_filter(index):
    skip_if_redis_version_below(index.client, "7.2.0")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]
    filter_expression = Geo("location") == GeoRadius(-122.4194, 37.7749, 1000, "m")

    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        filter_expression=filter_expression,
        return_fields=return_fields,
    )

    results = index.query(hybrid_query)
    assert len(results) == 3
    for result in results:
        assert result["location"] is not None


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_aggregate_query_alpha(index, alpha):
    skip_if_redis_version_below(index.client, "7.2.0")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"

    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        alpha=alpha,
    )

    results = index.query(hybrid_query)
    assert len(results) == 7
    for result in results:
        score = alpha * float(result["vector_similarity"]) + (1 - alpha) * float(
            result["text_score"]
        )
        assert (
            float(result["hybrid_score"]) - score <= 0.0001
        )  # allow for small floating point error


def test_aggregate_query_stopwords(index):
    skip_if_redis_version_below(index.client, "7.2.0")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    alpha = 0.5

    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        alpha=alpha,
        stopwords=["medical", "expertise"],
    )

    query_string = hybrid_query._build_query_string()

    assert "medical" not in query_string
    assert "expertize" not in query_string

    results = index.query(hybrid_query)
    assert len(results) == 7
    for result in results:
        score = alpha * float(result["vector_similarity"]) + (1 - alpha) * float(
            result["text_score"]
        )
        assert (
            float(result["hybrid_score"]) - score <= 0.0001
        )  # allow for small floating point error


def test_aggregate_query_with_text_filter(index):
    skip_if_redis_version_below(index.client, "7.2.0")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    filter_expression = Text(text_field) == ("medical")

    # make sure we can still apply filters to the same text field we are querying
    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        alpha=0.5,
        filter_expression=filter_expression,
        return_fields=["job", "description"],
    )

    results = index.query(hybrid_query)
    assert len(results) == 2
    for result in results:
        assert "medical" in result[text_field].lower()

    filter_expression = (Text(text_field) == ("medical")) & (
        (Text(text_field) != ("research"))
    )
    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        alpha=0.5,
        filter_expression=filter_expression,
        return_fields=["description"],
    )

    results = index.query(hybrid_query)
    assert len(results) == 2
    for result in results:
        assert "medical" in result[text_field].lower()
        assert "research" not in result[text_field].lower()
