import asyncio
import time
from pprint import pprint

import numpy as np
import pandas as pd
import pytest

from redisvl.index import SearchIndex
from redisvl.load import concurrent_store_as_hash
from redisvl.query import create_vector_query
from redisvl.utils.connection import get_async_redis_connection

data = pd.DataFrame(
    {
        "users": ["john", "mary", "joe"],
        "age": [1, 2, 3],
        "job": ["engineer", "doctor", "dentist"],
        "credit_score": ["high", "low", "medium"],
        "user_embedding": [
            np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes(),
            np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes(),
            np.array([0.9, 0.9, 0.1], dtype=np.float32).tobytes(),
        ],
    }
)

schema = {
    "index": {
        "name": "user_index",
        "prefix": "user:",
        "key_field": "users",
        "storage_type": "hash",
    },
    "fields": {
        "tag": {"credit_score": {}},
        "text": {"job": {}},
        "numeric": {"age": {}},
        "vector": {
            "user_embedding": {
                "dims": 3,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            }
        },
    },
}


@pytest.mark.asyncio
async def test_simple(async_redis):
    index = SearchIndex.from_dict(async_redis, schema)

    await concurrent_store_as_hash(
        data.to_dict(orient="records"),
        5,
        index.key_field,
        index.prefix,
        index.redis_conn,
    )
    await index.create()
    # add assertions here

    # wait for indexing to happen on server side
    time.sleep(1)

    query = create_vector_query(
        ["users", "age", "job", "credit_score", "vector_score"],
        number_of_results=3,
        vector_field_name="user_embedding",
    )

    query_vector = np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes()
    results = await async_redis.ft(index.index_name).search(
        query, query_params={"vector": query_vector}
    )

    # make sure correct users returned
    # users = list(results.docs)
    # print(len(users))
    users = [doc for doc in results.docs]
    assert users[0].users in ["john", "mary"]
    assert users[1].users in ["john", "mary"]

    # make sure vector scores are correct
    # query vector and first two are the same vector.
    # third is different (hence should be positive difference)
    assert float(users[0].vector_score) == 0.0
    assert float(users[1].vector_score) == 0.0
    assert float(users[2].vector_score) > 0

    print()
    for doc in results.docs:
        print("Score:", doc.vector_score)
        pprint(doc)

    await index.delete()
