from pprint import pprint

import numpy as np
import pytest

from redisvl.index import SearchIndex
from redisvl.query import NumericFilter, TagFilter, VectorQuery

data = [
    {
        "user": "john",
        "age": 18,
        "job": "engineer",
        "credit_score": "high",
        "user_embedding": np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "derrick",
        "age": 14,
        "job": "doctor",
        "credit_score": "low",
        "user_embedding": np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "nancy",
        "age": 94,
        "job": "doctor",
        "credit_score": "high",
        "user_embedding": np.array([0.7, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "tyler",
        "age": 100,
        "job": "engineer",
        "credit_score": "high",
        "user_embedding": np.array([0.1, 0.4, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "tim",
        "age": 12,
        "job": "dermatologist",
        "credit_score": "high",
        "user_embedding": np.array([0.4, 0.4, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "taimur",
        "age": 15,
        "job": "CEO",
        "credit_score": "low",
        "user_embedding": np.array([0.6, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "joe",
        "age": 35,
        "job": "dentist",
        "credit_score": "medium",
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
        return_fields=["user", "credit_score", "age", "job"],
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
    t = TagFilter("credit_score", "high")
    v = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        hybrid_filter=t,
    )

    results = index.search(v.query, query_params=v.params)
    assert len(results.docs) == 4


def test_simple_numeric_filter(index):
    # (@age:[18 101])=>[KNN 10 @user_embedding $vector AS vector_distance]
    n = NumericFilter("age", 18, 100)
    v = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        hybrid_filter=n,
    )

    results = index.search(v.query, query_params=v.params)
    assert len(results.docs) == 4


def test_numeric_filter_exclusive(index):
    n = NumericFilter("age", 18, 100, min_exclusive=True)
    v = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        hybrid_filter=n,
    )

    results = index.search(v.query, query_params=v.params)
    assert len(results.docs) == 3

    n_both_exclusive = NumericFilter(
        "age", 18, 100, min_exclusive=True, max_exclusive=True
    )
    v.set_filter(n_both_exclusive)
    results = index.search(v.query, query_params=v.params)
    assert len(results.docs) == 2


def test_combinations(index):
    # (@age:[18 100] @credit_score:{high})=>[KNN 10 @user_embedding $vector AS vector_distance]
    t = TagFilter("credit_score", "high")
    n = NumericFilter("age", 18, 100)
    t += n
    v = VectorQuery(
        [0.1, 0.1, 0.5],
        "user_embedding",
        return_fields=["user", "credit_score", "age", "job"],
        hybrid_filter=t,
    )

    results = index.search(v.query, query_params=v.params)
    for doc in results.docs:
        assert doc.credit_score == "high"
        assert 18 <= int(doc.age) <= 100
    assert len(results.docs) == 3

    # (@credit_score:{high} -@age:[18 100])=>[KNN 10 @user_embedding $vector AS vector_distance]
    t = TagFilter("credit_score", "high")
    n = NumericFilter("age", 18, 100)
    t -= n
    v.set_filter(t)

    results = index.search(v.query, query_params=v.params)
    for doc in results.docs:
        assert doc.credit_score == "high"
        assert int(doc.age) not in range(18, 101)
    assert len(results.docs) == 1

    # (@credit_score:{high} | @age:[18 100])=>[KNN 10 @user_embedding $vector AS vector_distance]
    t = TagFilter("credit_score", "high")
    n = NumericFilter("age", 18, 100)
    t &= n
    v.set_filter(t)

    results = index.search(v.query, query_params=v.params)
    for doc in results.docs:
        assert (doc.credit_score == "high") or (18 <= int(doc.age) <= 100)
    assert len(results.docs) == 5

    # (@credit_score:{high} ~@age:[18 100])=>[KNN 10 @user_embedding $vector AS vector_distance]
    t = TagFilter("credit_score", "high")
    n = NumericFilter("age", 18, 100)
    t ^= n
    v.set_filter(t)

    results = index.search(v.query, query_params=v.params)
    for doc in results.docs:
        assert doc.credit_score == "high"
    assert len(results.docs) == 4
