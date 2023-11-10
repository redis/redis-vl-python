import pytest
import numpy as np
from pprint import pprint

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.utils import array_to_buffer


data = [
    {
        "id": 1,
        "user": "john",
        "age": 1,
        "job": "engineer",
        "credit_score": "high",
        "user_embedding": [0.1, 0.1, 0.5],
    },
    {
        "id": 2,
        "user": "mary",
        "age": 2,
        "job": "doctor",
        "credit_score": "low",
        "user_embedding": [0.1, 0.1, 0.5],
    },
    {
        "id": 3,
        "user": "joe",
        "age": 3,
        "job": "dentist",
        "credit_score": "medium",
        "user_embedding": [0.9, 0.9, 0.1],
    },
]

hash_schema = {
    "index": {
        "name": "user_index_hash",
        "prefix": "users_hash",
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

json_schema = {
    "index": {
        "name": "user_index_json",
        "prefix": "users_json",
        "storage_type": "json",
    },
    "fields": {
        "tag": [{"name": "$.credit_score", "as_name": "credit_score"},
                {"name": "$.user", "as_name": "user"}],
        "text": [{"name": "$.job", "as_name": "job"}],
        "numeric": [{"name": "$.age", "as_name": "age"}],
        "vector": [
            {
                "name": "$.user_embedding",
                "as_name": "user_embedding",
                "dims": 3,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            }
        ],
    },
}

@pytest.mark.parametrize("schema", [hash_schema, json_schema])
def test_simple(client, schema):
    index = SearchIndex.from_dict(schema)
    # assign client (only for testing)
    index.set_client(client)
    # create the index
    index.create(overwrite=True)

    # Prepare and load the data based on storage type
    def hash_preprocess(item: dict) -> dict:
        return {**item, "user_embedding": array_to_buffer(item["user_embedding"])}
    if index.storage_type == "hash":
        index.load(data, preprocess=hash_preprocess)
    else:
        # Load the prepared data into the index
        index.load(data)

    query = VectorQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "age", "job", "credit_score"],
        num_results=3,
    )

    results = index.search(query.query, query_params=query.params)
    results_2 = index.query(query)
    assert len(results.docs) == len(results_2)

    # make sure correct users returned
    # users = list(results.docs)
    # print(len(users))
    users = [doc for doc in results.docs]
    pprint(users)
    assert users[0].user in ["john", "mary"]
    assert users[1].user in ["john", "mary"]

    # make sure vector scores are correct
    # query vector and first two are the same vector.
    # third is different (hence should be positive difference)
    assert float(users[0].vector_distance) == 0.0
    assert float(users[1].vector_distance) == 0.0
    assert float(users[2].vector_distance) > 0

    print()
    for doc in results.docs:
        print("Score:", doc.vector_distance)
        pprint(doc)

    index.delete()
