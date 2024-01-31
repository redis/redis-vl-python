import pytest

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import StorageType

fields_spec = [
    {"name": "credit_score", "type": "tag"},
    {"name": "user", "type": "tag"},
    {"name": "job", "type": "text"},
    {"name": "age", "type": "numeric"},
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
]

hash_schema = {
    "index": {
        "name": "user_index_hash",
        "prefix": "users_hash",
        "storage_type": "hash",
    },
    "fields": fields_spec,
}

json_schema = {
    "index": {
        "name": "user_index_json",
        "prefix": "users_json",
        "storage_type": "json",
    },
    "fields": fields_spec,
}


@pytest.mark.parametrize("schema", [hash_schema, json_schema])
def test_simple(client, schema, sample_data):
    index = SearchIndex.from_dict(schema)
    # assign client (only for testing)
    index.set_client(client)
    # create the index
    index.create(overwrite=True, drop=True)

    # Prepare and load the data based on storage type
    def hash_preprocess(item: dict) -> dict:
        return {**item, "user_embedding": array_to_buffer(item["user_embedding"])}

    if index.storage_type == StorageType.HASH:
        index.load(sample_data, preprocess=hash_preprocess)
    else:
        index.load(sample_data)

    return_fields = ["user", "age", "job", "credit_score"]
    query = VectorQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=return_fields,
        num_results=3,
    )

    results = index.search(query.query, query_params=query.params)
    results_2 = index.query(query)
    assert len(results.docs) == len(results_2)

    # make sure correct users returned
    # users = list(results.docs)
    # print(len(users))
    users = [doc for doc in results.docs]
    assert users[0].user in ["john", "mary"]
    assert users[1].user in ["john", "mary"]

    # make sure vector scores are correct
    # query vector and first two are the same vector.
    # third is different (hence should be positive difference)
    assert float(users[0].vector_distance) == 0.0
    assert float(users[1].vector_distance) == 0.0
    assert float(users[2].vector_distance) > 0

    for doc1, doc2 in zip(results.docs, results_2):
        for field in return_fields:
            assert getattr(doc1, field) == doc2[field]

    index.delete()
