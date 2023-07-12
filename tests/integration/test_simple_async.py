import time
from pprint import pprint

import numpy as np
import pandas as pd
import pytest

from redisvl.index import AsyncSearchIndex
from redisvl.query import create_vector_query

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
query_vector = np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes()

schema = {
    "index": {
        "name": "user_index",
        "prefix": "user:",
        "key_field": "users",
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


@pytest.mark.asyncio
async def test_simple(async_client):
    index = AsyncSearchIndex.from_dict(schema)
    # assign client (only for testing)
    index.set_client(async_client)
    # create the index
    await index.create()

    # load data into the index in Redis
    records = data.to_dict("records")
    await index.load(records)

    # wait for async index to create
    time.sleep(1)

    query = create_vector_query(
        ["users", "age", "job", "credit_score", "vector_score"],
        number_of_results=3,
        vector_field_name="user_embedding",
    )

    results = await index.search(query, query_params={"vector": query_vector})

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
