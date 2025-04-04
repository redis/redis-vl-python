from datetime import timedelta

import pytest
from redis.commands.search.result import Result

from redisvl.index import SearchIndex
from redisvl.query import (
    CountQuery,
    FilterQuery,
    TextQuery,
    VectorQuery,
    VectorRangeQuery,
)
from redisvl.query.filter import (
    FilterExpression,
    Geo,
    GeoRadius,
    Num,
    Tag,
    Text,
    Timestamp,
)
from redisvl.query.query import VectorRangeQuery
from redisvl.redis.utils import array_to_buffer

# TODO expand to multiple schema types and sync + async


@pytest.fixture
def vector_query():
    return VectorQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_score=True,
        return_fields=[
            "user",
            "credit_score",
            "age",
            "job",
            "location",
            "last_updated",
        ],
    )


@pytest.fixture
def sorted_vector_query():
    return VectorQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=[
            "user",
            "credit_score",
            "age",
            "job",
            "location",
            "last_updated",
        ],
        sort_by="age",
    )


@pytest.fixture
def normalized_vector_query():
    return VectorQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        normalize_vector_distance=True,
        return_score=True,
        return_fields=[
            "user",
            "credit_score",
            "age",
            "job",
            "location",
            "last_updated",
        ],
    )


@pytest.fixture
def filter_query():
    return FilterQuery(
        return_fields=[
            "user",
            "credit_score",
            "age",
            "job",
            "location",
            "last_updated",
        ],
        filter_expression=Tag("credit_score") == "high",
    )


@pytest.fixture
def sorted_filter_query():
    return FilterQuery(
        return_fields=[
            "user",
            "credit_score",
            "age",
            "job",
            "location",
            "last_updated",
        ],
        filter_expression=Tag("credit_score") == "high",
        sort_by="age",
    )


@pytest.fixture
def normalized_range_query():
    return VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        normalize_vector_distance=True,
        return_score=True,
        return_fields=["user", "credit_score", "age", "job", "location"],
        distance_threshold=0.2,
    )


@pytest.fixture
def range_query():
    return VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        distance_threshold=0.2,
    )


@pytest.fixture
def sorted_range_query():
    return VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        distance_threshold=0.2,
        sort_by="age",
    )


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
                {"name": "description", "type": "text"},
                {"name": "credit_score", "type": "tag"},
                {"name": "job", "type": "text"},
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


@pytest.fixture
def L2_index(sample_data, redis_url):
    # construct a search index from the schema
    index = SearchIndex.from_dict(
        {
            "index": {
                "name": "L2_index",
                "prefix": "L2_index",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "credit_score", "type": "tag"},
                {"name": "job", "type": "text"},
                {"name": "age", "type": "numeric"},
                {"name": "last_updated", "type": "numeric"},
                {"name": "location", "type": "geo"},
                {
                    "name": "user_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 3,
                        "distance_metric": "L2",
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


def test_search_and_query(index):
    # *=>[KNN 7 @user_embedding $vector AS vector_distance]
    v = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        num_results=7,
    )
    results = index.search(v.query, query_params=v.params)
    assert isinstance(results, Result)
    assert len(results.docs) == 7
    for doc in results.docs:
        # ensure all return fields present
        assert doc.user in [
            "john",
            "derrick",
            "nancy",
            "tyler",
            "tim",
            "taimur",
            "joe",
            "mary",
        ]
        assert int(doc.age) in [18, 14, 94, 100, 12, 15, 35]
        assert doc.job in ["engineer", "doctor", "dermatologist", "CEO", "dentist"]
        assert doc.credit_score in ["high", "low", "medium"]

    processed_results = index.query(v)
    assert len(processed_results) == 7
    assert isinstance(processed_results[0], dict)
    result = results.docs[0].__dict__
    result.pop("payload")
    assert processed_results[0] == results.docs[0].__dict__


def test_range_query(index):
    r = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        distance_threshold=0.2,
        num_results=7,
    )
    results = index.query(r)
    for result in results:
        assert float(result["vector_distance"]) <= 0.2
    assert len(results) == 4
    assert r.distance_threshold == 0.2

    r.set_distance_threshold(0.1)
    assert r.distance_threshold == 0.1
    results = index.query(r)
    for result in results:
        assert float(result["vector_distance"]) <= 0.1
    assert len(results) == 2


def test_count_query(index, sample_data):
    c = CountQuery(FilterExpression("*"))
    results = index.query(c)
    assert results == len(sample_data)

    c = CountQuery(Tag("credit_score") == "high")
    results = index.query(c)
    assert results == 4


def search(
    query,
    index,
    _filter,
    expected_count,
    credit_check=None,
    age_range=None,
    location=None,
    distance_threshold=0.2,
    sort=False,
):
    """Utility function to test filters."""

    # set the new filter
    query.set_filter(_filter)
    print(str(query))

    results = index.search(query.query, query_params=query.params)

    # check for tag filter correctness
    if credit_check:
        for doc in results.docs:
            assert doc.credit_score == credit_check

    # check for numeric filter correctness
    if age_range:
        for doc in results.docs:
            if len(age_range) == 3:
                assert int(doc.age) != age_range[2]
            elif age_range[1] < age_range[0]:
                assert (int(doc.age) <= age_range[0]) or (int(doc.age) >= age_range[1])
            else:
                assert age_range[0] <= int(doc.age) <= age_range[1]

    # check for geographic filter correctness
    if location:
        for doc in results.docs:
            assert doc.location == location

    # if range query, test results by distance threshold
    if isinstance(query, VectorRangeQuery):
        for doc in results.docs:
            print(doc.vector_distance)
            assert float(doc.vector_distance) <= distance_threshold

    # otherwise check by expected count.
    else:
        assert len(results.docs) == expected_count

    # check results are in sorted order
    if sort:
        if isinstance(query, VectorRangeQuery):
            assert [int(doc.age) for doc in results.docs] == [12, 14, 18, 100]
        else:
            assert [int(doc.age) for doc in results.docs] == [
                12,
                14,
                15,
                18,
                35,
                94,
                100,
            ]


@pytest.fixture(
    params=["vector_query", "filter_query", "range_query"],
    ids=["VectorQuery", "FilterQuery", "VectorRangeQuery"],
)
def query(request):
    return request.getfixturevalue(request.param)


def test_filters(index, query, sample_datetimes):
    # Simple Tag Filter
    t = Tag("credit_score") == "high"
    search(query, index, t, 4, credit_check="high")

    # Multiple Tags
    t = Tag("credit_score") == ["high", "low"]
    search(query, index, t, 6)

    # Empty tag filter
    t = Tag("credit_score") == []
    search(query, index, t, 7)

    # Simple Numeric Filter
    n1 = Num("age") >= 18
    search(query, index, n1, 4, age_range=(18, 100))

    # intersection of rules
    n2 = (Num("age") >= 18) & (Num("age") < 100)
    search(query, index, n2, 3, age_range=(18, 99))

    # union
    n3 = (Num("age") < 18) | (Num("age") > 94)
    search(query, index, n3, 4, age_range=(95, 17))

    n4 = Num("age") != 18
    search(query, index, n4, 6, age_range=(0, 0, 18))

    # Geographic filters
    g = Geo("location") == GeoRadius(-122.4194, 37.7749, 1, unit="m")
    search(query, index, g, 3, location="-122.4194,37.7749")

    g = Geo("location") != GeoRadius(-122.4194, 37.7749, 1, unit="m")
    search(query, index, g, 4, location="-110.0839,37.3861")

    # Text filters
    t = Text("job") == "engineer"
    search(query, index, t, 2)

    t = Text("job") != "engineer"
    search(query, index, t, 5)

    t = Text("job") % "enginee*"
    search(query, index, t, 2)

    t = Text("job") % "engine*|doctor"
    search(query, index, t, 4)

    t = Text("job") % "%%engine%%"
    search(query, index, t, 2)

    # Test empty filters
    t = Text("job") % ""
    search(query, index, t, 7)

    # Timestamps
    ts = Timestamp("last_updated") > sample_datetimes["mid"]
    search(query, index, ts, 2)

    ts = Timestamp("last_updated") >= sample_datetimes["mid"]
    search(query, index, ts, 5)

    ts = Timestamp("last_updated") < sample_datetimes["high"]
    search(query, index, ts, 5)

    ts = Timestamp("last_updated") <= sample_datetimes["mid"]
    search(query, index, ts, 5)

    ts = Timestamp("last_updated") == sample_datetimes["mid"]
    search(query, index, ts, 3)

    ts = (Timestamp("last_updated") == sample_datetimes["low"]) | (
        Timestamp("last_updated") == sample_datetimes["high"]
    )
    search(query, index, ts, 4)

    # could drop between if we prefer union syntax
    ts = Timestamp("last_updated").between(
        sample_datetimes["low"] + timedelta(seconds=1),
        sample_datetimes["high"] - timedelta(seconds=1),
    )
    search(query, index, ts, 3)


def test_manual_string_filters(index, query):
    # Simple Tag Filter
    t = "@credit_score:{high}"
    search(query, index, t, 4, credit_check="high")

    # Multiple Tags
    t = "@credit_score:{high|low}"
    search(query, index, t, 6)

    # Simple Numeric Filter
    n1 = "@age:[18 +inf]"
    search(query, index, n1, 4, age_range=(18, 100))

    # intersection of rules
    n2 = "@age:[18 (100]"
    search(query, index, n2, 3, age_range=(18, 99))

    n3 = "(@age:[-inf (18] | @age:[(94 +inf])"
    search(query, index, n3, 4, age_range=(95, 17))

    n4 = "(-@age:[18 18])"
    search(query, index, n4, 6, age_range=(0, 0, 18))

    # Geographic filters
    g = "@location:[-122.4194 37.7749 1 m]"
    search(query, index, g, 3, location="-122.4194,37.7749")

    g = "(-@location:[-122.4194 37.7749 1 m])"
    search(query, index, g, 4, location="-110.0839,37.3861")

    # Text filters
    t = "@job:engineer"
    search(query, index, t, 2)

    t = "(-@job:engineer)"
    search(query, index, t, 5)

    t = "@job:enginee*"
    search(query, index, t, 2)

    t = "@job:(engine*|doctor)"
    search(query, index, t, 4)

    t = "@job:*engine*"
    search(query, index, t, 2)


def test_filter_combinations(index, query):
    # test combinations
    # intersection
    t = Tag("credit_score") == "high"
    text = Text("job") == "engineer"
    search(query, index, t & text, 2, credit_check="high")

    # union
    t = Tag("credit_score") == "high"
    text = Text("job") == "engineer"
    search(query, index, t | text, 4, credit_check="high")

    # union of negated expressions
    _filter = (Tag("credit_score") != "high") & (Text("job") != "engineer")
    search(query, index, _filter, 3)

    # geo + text
    g = Geo("location") == GeoRadius(-122.4194, 37.7749, 1, unit="m")
    text = Text("job") == "engineer"
    search(query, index, g & text, 1, location="-122.4194,37.7749")

    # geo + text
    g = Geo("location") != GeoRadius(-122.4194, 37.7749, 1, unit="m")
    text = Text("job") == "engineer"
    search(query, index, g & text, 1, location="-110.0839,37.3861")

    # num + text + geo
    n = (Num("age") >= 18) & (Num("age") < 100)
    t = Text("job") != "engineer"
    g = Geo("location") == GeoRadius(-122.4194, 37.7749, 1, unit="m")
    search(query, index, n & t & g, 1, age_range=(18, 99), location="-122.4194,37.7749")


def test_paginate_vector_query(index, vector_query, sample_data):
    batch_size = 2
    all_results = []
    for i, batch in enumerate(index.paginate(vector_query, batch_size), start=1):
        all_results.extend(batch)
        assert len(batch) <= batch_size

    expected_total_results = len(sample_data)
    expected_iterations = -(-expected_total_results // batch_size)  # Ceiling division
    assert len(all_results) == expected_total_results
    assert i == expected_iterations


def test_paginate_filter_query(index, filter_query):
    batch_size = 3
    all_results = []
    for i, batch in enumerate(index.paginate(filter_query, batch_size), start=1):
        all_results.extend(batch)
        assert len(batch) <= batch_size

    expected_count = 4  # Adjust based on your filter
    expected_iterations = -(-expected_count // batch_size)  # Ceiling division
    assert len(all_results) == expected_count
    assert i == expected_iterations
    assert all(item["credit_score"] == "high" for item in all_results)


def test_paginate_range_query(index, range_query):
    batch_size = 1
    all_results = []
    for i, batch in enumerate(index.paginate(range_query, batch_size), start=1):
        all_results.extend(batch)
        assert len(batch) <= batch_size

    expected_count = 4  # Adjust based on your range query
    expected_iterations = -(-expected_count // batch_size)  # Ceiling division
    assert len(all_results) == expected_count
    assert i == expected_iterations
    assert all(float(item["vector_distance"]) <= 0.2 for item in all_results)


def test_sort_filter_query(index, sorted_filter_query):
    t = Text("job") % ""
    search(sorted_filter_query, index, t, 7, sort=True)


def test_sort_vector_query(index, sorted_vector_query):
    t = Text("job") % ""
    search(sorted_vector_query, index, t, 7, sort=True)


def test_sort_range_query(index, sorted_range_query):
    t = Text("job") % ""
    search(sorted_range_query, index, t, 7, sort=True)


def test_query_with_chunk_number_zero():
    doc_base_id = "8675309"
    file_id = "e9ffbac9ff6f67cc"
    chunk_num = 0

    filter_conditions = (
        (Tag("doc_base_id") == doc_base_id)
        & (Tag("file_id") == file_id)
        & (Num("chunk_number") == chunk_num)
    )

    expected_query_str = (
        "((@doc_base_id:{8675309} @file_id:{e9ffbac9ff6f67cc}) @chunk_number:[0 0])"
    )
    assert (
        str(filter_conditions) == expected_query_str
    ), "Query with chunk_number zero is incorrect"


def test_hybrid_policy_batches_mode(index, vector_query):
    """Test vector query with BATCHES hybrid policy."""
    # Create a filter
    t = Tag("credit_score") == "high"

    # Set hybrid policy to BATCHES
    vector_query.set_hybrid_policy("BATCHES")
    vector_query.set_batch_size(2)

    # Set the filter
    vector_query.set_filter(t)

    # Check query string
    assert "HYBRID_POLICY BATCHES BATCH_SIZE 2" in str(vector_query)

    # Execute query
    results = index.query(vector_query)

    # Check results - should have filtered to "high" credit scores
    assert len(results) > 0
    for result in results:
        assert result["credit_score"] == "high"


def test_hybrid_policy_adhoc_bf_mode(index, vector_query):
    """Test vector query with ADHOC_BF hybrid policy."""
    # Create a filter
    t = Tag("credit_score") == "high"

    # Set hybrid policy to ADHOC_BF
    vector_query.set_hybrid_policy("ADHOC_BF")

    # Set the filter
    vector_query.set_filter(t)

    # Check query string
    assert "HYBRID_POLICY ADHOC_BF" in str(vector_query)

    # Execute query
    results = index.query(vector_query)

    # Check results - should have filtered to "high" credit scores
    assert len(results) > 0
    for result in results:
        assert result["credit_score"] == "high"


def test_range_query_with_epsilon(index):
    """Integration test: Execute range query with epsilon parameter against Redis."""
    # Create a range query with epsilon
    epsilon_query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        distance_threshold=0.3,
        epsilon=0.5,  # Larger than default to get potentially more results
    )

    # Verify query string contains epsilon attribute
    query_string = str(epsilon_query)
    assert "$EPSILON: 0.5" in query_string

    # Verify epsilon property is set
    assert epsilon_query.epsilon == 0.5

    # Test setting epsilon
    epsilon_query.set_epsilon(0.1)
    assert epsilon_query.epsilon == 0.1
    assert "$EPSILON: 0.1" in str(epsilon_query)

    # Execute basic query without epsilon to ensure functionality
    basic_query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        distance_threshold=0.2,
    )

    results = index.query(basic_query)

    # Check results
    for result in results:
        assert float(result["vector_distance"]) <= 0.2


def test_range_query_with_filter_and_hybrid_policy(index):
    """Integration test: Test construction of a range query with filter and hybrid policy."""
    # Create a filter for high credit score
    credit_filter = Tag("credit_score") == "high"

    # Create a range query with filter and hybrid policy
    query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        filter_expression=credit_filter,
        distance_threshold=0.5,
        hybrid_policy="BATCHES",
        batch_size=2,
    )

    # Check query string and parameters
    query_string = str(query)
    assert "@credit_score:{high}" in query_string
    assert "HYBRID_POLICY" not in query_string
    assert query.hybrid_policy == "BATCHES"
    assert query.batch_size == 2
    assert query.params["HYBRID_POLICY"] == "BATCHES"
    assert query.params["BATCH_SIZE"] == 2

    # Execute basic query with filter but without hybrid policy
    basic_filter_query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        filter_expression=credit_filter,
        distance_threshold=0.5,
    )

    results = index.query(basic_filter_query)

    # Check results
    for result in results:
        assert result["credit_score"] == "high"
        assert float(result["vector_distance"]) <= 0.5


def test_query_normalize_cosine_distance(index, normalized_vector_query):

    res = index.query(normalized_vector_query)

    for r in res:
        assert 0 <= float(r["vector_distance"]) <= 1


def test_query_cosine_distance_un_normalized(index, vector_query):

    res = index.query(vector_query)

    assert any(float(r["vector_distance"]) > 1 for r in res)


def test_query_l2_distance_un_normalized(L2_index, vector_query):

    res = L2_index.query(vector_query)

    assert any(float(r["vector_distance"]) > 1 for r in res)


def test_query_l2_distance_normalized(L2_index, normalized_vector_query):

    res = L2_index.query(normalized_vector_query)

    for r in res:
        assert 0 <= float(r["vector_distance"]) <= 1


def test_range_query_normalize_cosine_distance(index, normalized_range_query):

    res = index.query(normalized_range_query)

    for r in res:
        assert 0 <= float(r["vector_distance"]) <= 1


def test_range_query_normalize_bad_input(index):
    with pytest.raises(ValueError):
        VectorRangeQuery(
            vector=[0.1, 0.1, 0.5],
            vector_field_name="user_embedding",
            normalize_vector_distance=True,
            return_score=True,
            return_fields=["user", "credit_score", "age", "job", "location"],
            distance_threshold=1.2,
        )


@pytest.mark.parametrize(
    "scorer", ["BM25", "TFIDF", "TFIDF.DOCNORM", "DISMAX", "DOCSCORE"]
)
def test_text_query(index, scorer):
    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]

    text_query = TextQuery(
        text=text,
        text_field_name=text_field,
        text_scorer=scorer,
        return_fields=return_fields,
    )
    results = index.query(text_query)

    assert len(results) == 4

    # make sure at least one word from the query is in the description
    for result in results:
        assert any(word in result[text_field] for word in text.split())


# test that text queryies work with filter expressions
def test_text_query_with_filter(index):
    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    return_fields = ["user", "credit_score", "age", "job", "location", "description"]
    filter_expression = (Tag("credit_score") == ("high")) & (Num("age") > 30)
    scorer = "TFIDF"

    text_query = TextQuery(
        text=text,
        text_field_name=text_field,
        text_scorer=scorer,
        filter_expression=filter_expression,
        return_fields=return_fields,
    )
    results = index.query(text_query)
    assert len(results) == 2
    for result in results:
        assert any(word in result[text_field] for word in text.split())
        assert result["credit_score"] == "high"
        assert int(result["age"]) > 30


# test that text queryies workt with text filter expressions on the same text field
def test_text_query_with_text_filter(index):
    text = "a medical professional with expertise in lung cancer"
    text_field = "description"
    scorer = "TFIDF.DOCNORM"
    return_fields = ["age", "job", "description"]
    filter_expression = Text(text_field) == ("medical")

    text_query = TextQuery(
        text=text,
        text_field_name=text_field,
        text_scorer=scorer,
        filter_expression=filter_expression,
        return_fields=return_fields,
    )
    results = index.query(text_query)
    assert len(results) == 2
    for result in results:
        assert any(word in result[text_field] for word in text.split())
        assert "medical" in result[text_field]

    filter_expression = Text(text_field) != ("research")

    text_query = TextQuery(
        text=text,
        text_field_name=text_field,
        text_scorer=scorer,
        filter_expression=filter_expression,
        return_fields=return_fields,
    )

    results = index.query(text_query)
    assert len(results) == 3
    for result in results:
        assert any(word in result[text_field] for word in text.split())
        assert "research" not in result[text_field]
