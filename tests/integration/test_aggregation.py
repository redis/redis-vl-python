import pytest

from redisvl.index import SearchIndex
from redisvl.query import HybridQuery, MultiVectorQuery, Vector
from redisvl.query.filter import FilterExpression, Geo, GeoRadius, Num, Tag, Text
from redisvl.redis.utils import array_to_buffer
from tests.conftest import skip_if_redis_version_below


@pytest.fixture
def index(multi_vector_data, redis_url, worker_id):

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
            "image_embedding": array_to_buffer(item["image_embedding"], "float32"),
            "audio_embedding": array_to_buffer(item["audio_embedding"], "float64"),
        }

    index.load(multi_vector_data, preprocess=hash_preprocess)

    # run the test
    yield index

    # clean up
    index.delete(drop=True)


def test_hybrid_query(index):
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


def test_hybrid_query_with_filter(index):
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


def test_hybrid_query_with_geo_filter(index):
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
def test_hybrid_query_alpha(index, alpha):
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


def test_hybrid_query_stopwords(index):
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


def test_hybrid_query_with_text_filter(index):
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


@pytest.mark.parametrize("scorer", ["BM25", "BM25STD", "TFIDF", "TFIDF.DOCNORM"])
def test_hybrid_query_word_weights(index, scorer):
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
    )

    weighted_results = index.query(weighted_query)
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
    )

    unweighted_results = index.query(unweighted_query)

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
    )

    weighted_results = index.query(weighted_query)
    assert weighted_results != unweighted_results

    # test assigning weights on construction is equivalent to setting them on the query object
    new_query = HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        return_fields=return_fields,
        text_scorer=scorer,
        text_weights=None,
    )

    new_query.set_text_weights(weights)

    new_weighted_results = index.query(new_query)
    assert new_weighted_results == weighted_results


def test_multivector_query(index):
    skip_if_redis_version_below(index.client, "7.2.0")

    vector_vals = [[0.1, 0.1, 0.5], [0.3, 0.4, 0.7, 0.2, -0.3]]
    vector_fields = ["user_embedding", "image_embedding"]
    vectors = []
    for vector, field in zip(vector_vals, vector_fields):
        vectors.append(Vector(vector=vector, field_name=field))

    return_fields = ["user", "credit_score", "age", "job", "location", "description"]

    multi_query = MultiVectorQuery(
        vectors=vectors,
        return_fields=return_fields,
    )

    results = index.query(multi_query)
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

    multi_query = MultiVectorQuery(
        vectors=vectors,
        num_results=3,
    )

    results = index.query(multi_query)
    assert len(results) == 3
    assert (
        results[0]["combined_score"]
        >= results[1]["combined_score"]
        >= results[2]["combined_score"]
    )


def test_multivector_query_with_filter(index):
    skip_if_redis_version_below(index.client, "7.2.0")

    text_field = "description"
    vector_vals = [[0.1, 0.1, 0.5], [0.3, 0.4, 0.7, 0.2, -0.3]]
    vector_fields = ["user_embedding", "image_embedding"]
    filter_expression = Text(text_field) == ("medical")

    vectors = []
    for vector, field in zip(vector_vals, vector_fields):
        vectors.append(Vector(vector=vector, field_name=field))

    # make sure we can still apply filters to the same text field we are querying
    multi_query = MultiVectorQuery(
        vectors=vectors,
        filter_expression=filter_expression,
        return_fields=["job", "description"],
    )

    results = index.query(multi_query)
    assert len(results) == 2
    for result in results:
        assert "medical" in result[text_field].lower()

    filter_expression = (Text(text_field) == ("medical")) & (
        (Text(text_field) != ("research"))
    )
    multi_query = MultiVectorQuery(
        vectors=vectors,
        filter_expression=filter_expression,
        return_fields=["description"],
    )

    results = index.query(multi_query)
    assert len(results) == 2
    for result in results:
        assert "medical" in result[text_field].lower()
        assert "research" not in result[text_field].lower()

    filter_expression = (Num("age") > 30) & ((Num("age") < 30))
    multi_query = MultiVectorQuery(
        vectors=vectors,
        filter_expression=filter_expression,
        return_fields=["description"],
    )

    results = index.query(multi_query)
    assert len(results) == 0


def test_multivector_query_with_geo_filter(index):
    skip_if_redis_version_below(index.client, "7.2.0")

    vector_vals = [[0.2, 0.4, 0.1], [0.1, 0.8, 0.3, -0.2, 0.3]]
    vector_fields = ["user_embedding", "image_embedding"]
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]
    filter_expression = Geo("location") == GeoRadius(-122.4194, 37.7749, 1000, "m")

    vectors = []
    for vector, field in zip(vector_vals, vector_fields):
        vectors.append(Vector(vector=vector, field_name=field))

    multi_query = MultiVectorQuery(
        vectors=vectors,
        filter_expression=filter_expression,
        return_fields=return_fields,
    )

    results = index.query(multi_query)
    assert len(results) == 3
    for result in results:
        assert result["location"] is not None


def test_multivector_query_weights(index):
    skip_if_redis_version_below(index.client, "7.2.0")

    vector_vals = [[0.1, 0.2, 0.5], [0.3, 0.4, 0.7, 0.2, -0.3]]
    vector_fields = ["user_embedding", "image_embedding"]
    return_fields = [
        "distance_0",
        "distance_1",
        "score_0",
        "score_1",
        "user_embedding",
        "image_embedding",
    ]

    vectors = []
    for vector, field in zip(vector_vals, vector_fields):
        vectors.append(Vector(vector=vector, field_name=field))

    # changing the weights does indeed change the result order
    multi_query_1 = MultiVectorQuery(
        vectors=vectors,
        return_fields=return_fields,
    )
    results_1 = index.query(multi_query_1)

    weights = [0.2, 0.9]
    vectors = []
    for vector, field, weight in zip(vector_vals, vector_fields, weights):
        vectors.append(Vector(vector=vector, field_name=field, weight=weight))

    multi_query_2 = MultiVectorQuery(
        vectors=vectors,
        return_fields=return_fields,
    )
    results_2 = index.query(multi_query_2)

    assert results_1 != results_2

    for i in range(1, len(results_1)):
        assert results_1[i]["combined_score"] <= results_1[i - 1]["combined_score"]

    for i in range(1, len(results_2)):
        assert results_2[i]["combined_score"] <= results_2[i - 1]["combined_score"]

    # weights can be negative, 0.0, or greater than 1.0
    weights = [-5.2, 0.0]
    vectors = []
    for vector, field, weight in zip(vector_vals, vector_fields, weights):
        vectors.append(Vector(vector=vector, field_name=field, weight=weight))

    multi_query = MultiVectorQuery(
        vectors=vectors,
        return_fields=return_fields,
    )

    results = index.query(multi_query)
    assert results
    for r in results:
        score = float(r["score_0"]) * weights[0]
        assert (
            float(r["combined_score"]) - score <= 0.0001
        )  # allow for small floating point error

    # verify we're doing the combined score math correctly
    weights = [-1.322, 0.851]
    vectors = []
    for vector, field, weight in zip(vector_vals, vector_fields, weights):
        vectors.append(Vector(vector=vector, field_name=field, weight=weight))

    multi_query = MultiVectorQuery(
        vectors=vectors,
        return_fields=return_fields,
    )

    results = index.query(multi_query)
    assert results
    for r in results:
        score = float(r["score_0"]) * weights[0] + float(r["score_1"]) * weights[1]
        assert (
            float(r["combined_score"]) - score <= 0.0001
        )  # allow for small floating point error


def test_multivector_query_datatypes(index):
    skip_if_redis_version_below(index.client, "7.2.0")

    vector_vals = [[0.1, 0.2, 0.5], [1.2, 0.3, -0.4, 0.7, 0.2, -0.3]]
    vector_fields = ["user_embedding", "audio_embedding"]
    dtypes = ["float32", "float64"]
    return_fields = [
        "distance_0",
        "distance_1",
        "score_0",
        "score_1",
        "user_embedding",
        "audio_embedding",
    ]

    vectors = []
    for vector, field, dtype in zip(vector_vals, vector_fields, dtypes):
        vectors.append(Vector(vector=vector, field_name=field, dtype=dtype))

    multi_query = MultiVectorQuery(
        vectors=vectors,
        return_fields=return_fields,
    )
    results = index.query(multi_query)

    for i in range(1, len(results)):
        assert results[i]["combined_score"] <= results[i - 1]["combined_score"]

    # verify we're doing the combined score math correctly
    weights = [-1.322, 0.851]
    vectors = []
    for vector, field, weight, dtype in zip(
        vector_vals, vector_fields, weights, dtypes
    ):
        vectors.append(
            Vector(vector=vector, field_name=field, weight=weight, dtype=dtype)
        )

    multi_query = MultiVectorQuery(
        vectors=vectors,
        return_fields=return_fields,
    )

    results = index.query(multi_query)
    assert results
    for r in results:
        score = float(r["score_0"]) * weights[0] + float(r["score_1"]) * weights[1]
        assert (
            float(r["combined_score"]) - score <= 0.0001
        )  # allow for small floating point error


def test_multivector_query_mixed_index(index):
    # test that we can do multi vector queries on indices with both a 'flat' and 'hnsw' index
    skip_if_redis_version_below(index.client, "7.2.0")
    try:
        index.schema.remove_field("audio_embedding")
        index.schema.add_field(
            {
                "name": "audio_embedding",
                "type": "vector",
                "attrs": {
                    "dims": 6,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float64",
                },
            },
        )

    except:
        pytest.skip("Required Redis modules not available or version too low")

    vector_vals = [[0.1, 0.2, 0.5], [1.2, 0.3, -0.4, 0.7, 0.2, -0.3]]
    vector_fields = ["user_embedding", "audio_embedding"]
    dtypes = ["float32", "float64"]
    return_fields = [
        "distance_0",
        "distance_1",
        "score_0",
        "score_1",
        "user_embedding",
        "audio_embedding",
    ]

    vectors = []
    for vector, field, dtype in zip(vector_vals, vector_fields, dtypes):
        vectors.append(Vector(vector=vector, field_name=field, dtype=dtype))

    multi_query = MultiVectorQuery(
        vectors=vectors,
        return_fields=return_fields,
    )
    results = index.query(multi_query)

    for i in range(1, len(results)):
        assert results[i]["combined_score"] <= results[i - 1]["combined_score"]

    # verify we're doing the combined score math correctly
    weights = [-1.322, 0.851]
    vectors = []
    for vector, field, dtype, weight in zip(
        vector_vals, vector_fields, dtypes, weights
    ):
        vectors.append(
            Vector(vector=vector, field_name=field, dtype=dtype, weight=weight)
        )

    multi_query = MultiVectorQuery(
        vectors=vectors,
        return_fields=return_fields,
    )

    results = index.query(multi_query)
    assert results
    for r in results:
        score = float(r["score_0"]) * weights[0] + float(r["score_1"]) * weights[1]
        assert (
            float(r["combined_score"]) - score <= 0.0001
        )  # allow for small floating point error
