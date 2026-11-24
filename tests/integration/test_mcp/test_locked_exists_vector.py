"""A locked ``exists`` filter on a vector field, checked against live Redis.

``validate_profile_against_schema`` accepts ``exists`` on a vector field only
when the field declares ``index_missing``, on the grounds that Redis then answers
``ismissing()`` for it. These tests pin that assumption: if ``FT.CREATE`` ever
stopped carrying ``INDEXMISSING`` for a vector field, the validator would still
pass the profile and every call would fail as a retryable backend error.
"""

import numpy as np
import pytest

from redisvl.index import SearchIndex
from redisvl.mcp.filters import parse_filter
from redisvl.query import FilterQuery, VectorQuery
from redisvl.redis.connection import supports_svs
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema

DIMS = 4
EMBEDDING = [0.1, 0.2, 0.3, 0.4]


@pytest.fixture(
    params=[
        pytest.param({"algorithm": "flat"}, id="flat"),
        pytest.param({"algorithm": "hnsw"}, id="hnsw"),
        pytest.param({"algorithm": "svs-vamana"}, id="svs-vamana"),
    ]
)
def vector_attrs(request, client):
    if request.param["algorithm"] == "svs-vamana" and not supports_svs(client):
        pytest.skip("SVS-VAMANA requires Redis 8.2+ with SVS support")
    return request.param


@pytest.fixture(params=["hash", "json"])
def storage_type(request):
    return request.param


@pytest.fixture
def index_with_missing_embedding(
    redis_url, redis_test_name, vector_attrs, storage_type
):
    name = redis_test_name("locked_exists_vector")
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": name,
                "prefix": name,
                "storage_type": storage_type,
            },
            "fields": [
                {"name": "doc_id", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        **vector_attrs,
                        "dims": DIMS,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                        "index_missing": True,
                    },
                },
            ],
        }
    )
    index = SearchIndex(schema=schema, redis_url=redis_url)
    index.create(overwrite=True, drop=True)

    embedding = (
        array_to_buffer(EMBEDDING, "float32") if storage_type == "hash" else EMBEDDING
    )
    index.load(
        [
            {"doc_id": "with-embedding", "embedding": embedding},
            {"doc_id": "without-embedding"},
        ],
        id_field="doc_id",
    )

    yield index

    index.delete(drop=True)


def _locked_exists_filter(index):
    # The same parser a profile's lock.filter goes through at startup.
    return parse_filter({"field": "embedding", "op": "exists"}, index.schema)


def test_locked_exists_on_vector_field_as_filter(index_with_missing_embedding):
    index = index_with_missing_embedding
    locked = _locked_exists_filter(index)
    assert str(locked) == "(-ismissing(@embedding))"

    results = index.query(
        FilterQuery(filter_expression=locked, return_fields=["doc_id"])
    )

    assert [doc["doc_id"] for doc in results] == ["with-embedding"]


def test_locked_exists_on_vector_field_as_knn_prefilter(index_with_missing_embedding):
    index = index_with_missing_embedding
    query = VectorQuery(
        vector=np.array(EMBEDDING, dtype=np.float32).tolist(),
        vector_field_name="embedding",
        return_fields=["doc_id"],
        filter_expression=_locked_exists_filter(index),
        num_results=10,
    )

    results = index.query(query)

    assert [doc["doc_id"] for doc in results] == ["with-embedding"]
