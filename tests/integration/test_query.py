from pprint import pprint

import numpy as np
import pytest

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Geo, GeoRadius, Num, Tag, Text

data = [
    {
        "user": "john",
        "age": 18,
        "job": "engineer",
        "credit_score": "high",
        "location": "-122.4194,37.7749",
        "user_embedding": np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "derrick",
        "age": 14,
        "job": "doctor",
        "credit_score": "low",
        "location": "-122.4194,37.7749",
        "user_embedding": np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "nancy",
        "age": 94,
        "job": "doctor",
        "credit_score": "high",
        "location": "-122.4194,37.7749",
        "user_embedding": np.array([0.7, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "tyler",
        "age": 100,
        "job": "engineer",
        "credit_score": "high",
        "location": "-110.0839,37.3861",
        "user_embedding": np.array([0.1, 0.4, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "tim",
        "age": 12,
        "job": "dermatologist",
        "credit_score": "high",
        "location": "-110.0839,37.3861",
        "user_embedding": np.array([0.4, 0.4, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "taimur",
        "age": 15,
        "job": "CEO",
        "credit_score": "low",
        "location": "-110.0839,37.3861",
        "user_embedding": np.array([0.6, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "joe",
        "age": 35,
        "job": "dentist",
        "credit_score": "medium",
        "location": "-110.0839,37.3861",
        "user_embedding": np.array([0.9, 0.9, 0.1], dtype=np.float32).tobytes(),
    },
]

schema = {
    "index": {
        "name": "user_index",
        "prefix": "v1",
        "storage_type": "hash",
    },
    "fields": {
        "tag": [{"name": "credit_score"}],
        "text": [{"name": "job"}],
        "numeric": [{"name": "age"}],
        "geo": [{"name": "location"}],
        "vector": [
            {
                "name": "user_embedding",
                "dims": 3,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            }
        ],
    },
}


@pytest.fixture(scope="module")
def index():
    # construct a search index from the schema
    index = SearchIndex.from_dict(schema)

    # connect to local redis instance
    index.connect("redis://localhost:6379")

    # create the index (no data yet)
    index.create(overwrite=True)

    index.load(data)

    # run the test
    yield index

    # clean up
    index.delete()


def test_simple(index):
    # *=>[KNN 7 @user_embedding $vector AS vector_distance]
    v = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        num_results=7,
    )
    results = index.search(v.query, query_params=v.params)
    assert len(results.docs) == 7
    for doc in results.docs:
        # ensure all return fields present
        assert doc.user in ["john", "derrick", "nancy", "tyler", "tim", "taimur", "joe"]
        assert int(doc.age) in [18, 14, 94, 100, 12, 15, 35]
        assert doc.job in ["engineer", "doctor", "dermatologist", "CEO", "dentist"]
        assert doc.credit_score in ["high", "low", "medium"]


def test_simple_tag_filter(index):
    # (@credit_score:{high})=>[KNN 10 @user_embedding $vector AS vector_distance]
    t = Tag("credit_score") == "high"
    v = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        hybrid_filter=t,
    )

    results = index.search(v.query, query_params=v.params)
    assert len(results.docs) == 4


def filter_test(
    index, _filter, expected_count, credit_check=None, age_range=None, location=None
):
    """Utility function to test filters"""
    v = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job", "location"],
        hybrid_filter=_filter,
    )
    # print(str(v) + "\n") # to print the query
    results = index.query(v)
    if credit_check:
        for doc in results.docs:
            assert doc.credit_score == credit_check
    if age_range:
        for doc in results.docs:
            if len(age_range) == 3:
                assert int(doc.age) != age_range[2]
            elif age_range[1] < age_range[0]:
                assert (int(doc.age) <= age_range[0]) or (int(doc.age) >= age_range[1])
            else:
                assert age_range[0] <= int(doc.age) <= age_range[1]
    if location:
        for doc in results.docs:
            assert doc.location == location
    assert len(results.docs) == expected_count


def test_filters(index):
    # Simple Tag Filter
    t = Tag("credit_score") == "high"
    filter_test(index, t, 4, credit_check="high")

    # Simple Numeric Filter
    n1 = Num("age") >= 18
    filter_test(index, n1, 4, age_range=(18, 100))

    # intersection of rules
    n2 = (Num("age") >= 18) & (Num("age") < 100)
    filter_test(index, n2, 3, age_range=(18, 99))

    # union
    n3 = (Num("age") < 18) | (Num("age") > 94)
    filter_test(index, n3, 4, age_range=(95, 17))

    n4 = Num("age") != 18
    filter_test(index, n4, 6, age_range=(0, 0, 18))

    # Geographic filters
    g = Geo("location") == GeoRadius(-122.4194, 37.7749, 1, unit="m")
    filter_test(index, g, 3, location="-122.4194,37.7749")

    g = Geo("location") != GeoRadius(-122.4194, 37.7749, 1, unit="m")
    filter_test(index, g, 4, location="-110.0839,37.3861")

    # Text filters
    t = Text("job") == "engineer"
    filter_test(index, t, 2)

    t = Text("job") != "engineer"
    filter_test(index, t, 5)

    t = Text("job") % "enginee*"
    filter_test(index, t, 2)


def test_filter_combinations(index):
    # test combinations
    # intersection
    t = Tag("credit_score") == "high"
    text = Text("job") == "engineer"
    filter_test(index, t & text, 2, credit_check="high")

    # union
    t = Tag("credit_score") == "high"
    text = Text("job") == "engineer"
    filter_test(index, t | text, 4, credit_check="high")

    # union of negated expressions
    _filter = (Tag("credit_score") != "high") & (Text("job") != "engineer")
    filter_test(index, _filter, 3)

    # geo + text
    g = Geo("location") == GeoRadius(-122.4194, 37.7749, 1, unit="m")
    text = Text("job") == "engineer"
    filter_test(index, g & text, 1, location="-122.4194,37.7749")

    # geo + text
    g = Geo("location") != GeoRadius(-122.4194, 37.7749, 1, unit="m")
    text = Text("job") == "engineer"
    filter_test(index, g & text, 1, location="-110.0839,37.3861")

    # num + text + geo
    n = (Num("age") >= 18) & (Num("age") < 100)
    t = Text("job") != "engineer"
    g = Geo("location") == GeoRadius(-122.4194, 37.7749, 1, unit="m")
    filter_test(index, n & t & g, 1, age_range=(18, 99), location="-122.4194,37.7749")
