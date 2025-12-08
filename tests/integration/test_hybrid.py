import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query.filter import Geo, GeoRadius, Num, Tag, Text
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema
from tests.conftest import (
    skip_if_redis_version_below,
    skip_if_redis_version_below_async,
)

try:
    from redisvl.query.hybrid import HybridQuery
except (ImportError, ModuleNotFoundError):
    HybridQuery = None  # type: ignore


@pytest.fixture
def index_schema(worker_id):
    return IndexSchema.from_dict(
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
                {
                    "name": "image_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 5,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
                {
                    "name": "audio_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 6,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float64",
                    },
                },
            ],
        }
    )


@pytest.fixture
def index(index_schema, multi_vector_data, redis_url):
    index = SearchIndex(schema=index_schema, redis_url=redis_url)

    # create the index (no data yet)
    index.create(overwrite=True)

    # prepare and load the data
    def hash_preprocess(item: dict) -> dict:
        return {
            **item,
            "user_embedding": array_to_buffer(item["user_embedding"], "float32"),
            "image_embedding": array_to_buffer(item["image_embedding"], "float32"),
            "audio_embedding": array_to_buffer(item["audio_embedding"], "float64"),
        }

    index.load(multi_vector_data, preprocess=hash_preprocess)

    # run the test
    yield index

    # clean up
    index.delete(drop=True)


@pytest.fixture
async def async_index(index_schema, multi_vector_data, async_client):
    index = AsyncSearchIndex(schema=index_schema, redis_client=async_client)
    await index.create(overwrite=True)

    def hash_preprocess(item: dict) -> dict:
        return {
            **item,
            "user_embedding": array_to_buffer(item["user_embedding"], "float32"),
            "image_embedding": array_to_buffer(item["image_embedding"], "float32"),
            "audio_embedding": array_to_buffer(item["audio_embedding"], "float64"),
        }

    await index.load(multi_vector_data, preprocess=hash_preprocess)
    yield index
    await index.delete(drop=True)


def test_hybrid_query(index):
    skip_if_redis_version_below(index.client, "8.4.0")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]

    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        yield_text_score_as="text_score",
        vector=vector,
        vector_field_name=vector_field,
        yield_vsim_score_as="vsim_score",
        combination_method="RRF",
        yield_combined_score_as="hybrid_score",
        return_fields=return_fields,
    )

    results = index.hybrid_search(hybrid_query)
    assert isinstance(results, list)
    assert len(results) == 10  # Server-side default for hybrid search
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
        combination_method="RRF",
        yield_combined_score_as="hybrid_score",
    )

    results = index.hybrid_search(hybrid_query)
    assert len(results) == 3
    assert (
        results[0]["hybrid_score"]
        >= results[1]["hybrid_score"]
        >= results[2]["hybrid_score"]
    )


def test_hybrid_query_with_filter(index):
    skip_if_redis_version_below(index.client, "8.4.0")

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
        text_filter_expression=filter_expression,
        vector_filter_expression=filter_expression,
        return_fields=return_fields,
    )

    results = index.hybrid_search(hybrid_query)
    assert len(results) == 2
    for result in results:
        assert result["credit_score"] == "high"
        assert int(result["age"]) > 30


def test_hybrid_query_with_geo_filter(index):
    skip_if_redis_version_below(index.client, "8.4.0")

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
        text_filter_expression=filter_expression,
        vector_filter_expression=filter_expression,
        return_fields=return_fields,
    )

    results = index.hybrid_search(hybrid_query)
    assert len(results) == 3
    for result in results:
        assert result["location"] is not None


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_hybrid_query_alpha(index, alpha):
    skip_if_redis_version_below(index.client, "8.4.0")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"

    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        combination_method="LINEAR",
        linear_alpha=alpha,
        yield_text_score_as="text_score",
        yield_vsim_score_as="vector_similarity",
        yield_combined_score_as="hybrid_score",
    )

    results = index.hybrid_search(hybrid_query)
    assert len(results) == 7
    for result in results:
        score = alpha * float(result["text_score"]) + (1 - alpha) * float(
            result["vector_similarity"]
        )
        assert (
            float(result["hybrid_score"]) - score <= 0.0001
        )  # allow for small floating point error


def test_hybrid_query_stopwords(index):
    skip_if_redis_version_below(index.client, "8.4.0")

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
        combination_method="LINEAR",
        linear_alpha=alpha,
        stopwords={"medical", "expertise"},
        yield_text_score_as="text_score",
        yield_vsim_score_as="vector_similarity",
        yield_combined_score_as="hybrid_score",
    )

    query_string = hybrid_query.query._search_query.query_string()
    assert "medical" not in query_string
    assert "expertise" not in query_string

    results = index.hybrid_search(hybrid_query)
    assert len(results) == 7
    for result in results:
        score = alpha * float(result["text_score"]) + (1 - alpha) * float(
            result["vector_similarity"]
        )
        assert (
            float(result["hybrid_score"]) - score <= 0.0001
        )  # allow for small floating point error


def test_hybrid_query_with_text_filter(index):
    skip_if_redis_version_below(index.client, "8.4.0")

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
        text_filter_expression=filter_expression,
        vector_filter_expression=filter_expression,
        combination_method="LINEAR",
        yield_combined_score_as="hybrid_score",
        return_fields=[text_field],
    )

    results = index.hybrid_search(hybrid_query)
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
        text_filter_expression=filter_expression,
        vector_filter_expression=filter_expression,
        combination_method="LINEAR",
        yield_combined_score_as="hybrid_score",
        return_fields=[text_field],
    )

    results = index.hybrid_search(hybrid_query)
    assert len(results) == 2
    for result in results:
        assert "medical" in result[text_field].lower()
        assert "research" not in result[text_field].lower()


@pytest.mark.parametrize("scorer", ["BM25STD", "TFIDF", "TFIDF.DOCNORM"])
def test_hybrid_query_word_weights(index, scorer):
    skip_if_redis_version_below(index.client, "8.4.0")

    text = "a medical professional with expertise in lung cancers"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["description"]

    weights = {"medical": 3.4, "cancers": 5}

    # test we can run a query with text weights
    weighted_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        return_fields=return_fields,
        text_scorer=scorer,
        text_weights=weights,
        yield_text_score_as="text_score",
    )

    weighted_results = index.hybrid_search(weighted_query)
    assert len(weighted_results) == 7

    # test that weights do change the scores on results
    unweighted_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        return_fields=return_fields,
        text_scorer=scorer,
        text_weights={},
        yield_text_score_as="text_score",
    )

    unweighted_results = index.hybrid_search(unweighted_query)

    for weighted, unweighted in zip(weighted_results, unweighted_results):
        for word in weights:
            if word in weighted["description"] or word in unweighted["description"]:
                assert float(weighted["text_score"]) > float(unweighted["text_score"])

    # test that weights do change the document score and order of results
    weights = {"medical": 5, "cancers": 3.4}  # switch the weights
    weighted_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        return_fields=return_fields,
        text_scorer=scorer,
        text_weights=weights,
        yield_text_score_as="text_score",
    )

    weighted_results = index.hybrid_search(weighted_query)
    assert weighted_results != unweighted_results


@pytest.mark.asyncio
async def test_hybrid_query_async(async_index):
    await skip_if_redis_version_below_async(async_index.client, "8.4.0")

    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]

    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        yield_text_score_as="text_score",
        vector=vector,
        vector_field_name=vector_field,
        yield_vsim_score_as="vsim_score",
        combination_method="RRF",
        yield_combined_score_as="hybrid_score",
        return_fields=return_fields,
    )

    results = await async_index.hybrid_search(hybrid_query)
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
        combination_method="RRF",
        yield_combined_score_as="hybrid_score",
    )

    results = await async_index.hybrid_search(hybrid_query)
    assert len(results) == 3
    assert (
        results[0]["hybrid_score"]
        >= results[1]["hybrid_score"]
        >= results[2]["hybrid_score"]
    )
