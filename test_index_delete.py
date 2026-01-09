schema = {
    "index": {
        "name": "user_simple",
        "prefix": "user_simple_docs",
    },
    "fields": [
        {"name": "user", "type": "tag"},
        {"name": "credit_score", "type": "tag"},
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
    ],
}

import numpy as np

data = [
    {
        "user": "john",
        "age": 1,
        "job": "engineer",
        "credit_score": "high",
        "user_embedding": np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "mary",
        "age": 2,
        "job": "doctor",
        "credit_score": "low",
        "user_embedding": np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "user": "joe",
        "age": 3,
        "job": "dentist",
        "credit_score": "medium",
        "user_embedding": np.array([0.9, 0.9, 0.1], dtype=np.float32).tobytes(),
    },
]

from redis import Redis

from redisvl.index import SearchIndex

client_7_4 = Redis.from_url("redis://localhost:6379")
index_7_4 = SearchIndex.from_dict(
    schema, redis_client=client_7_4, validate_on_load=True
)

index_7_4.create(overwrite=True)
index_7_4.load(data)
print(f"{index_7_4.info()['num_docs']=}")

client_8 = Redis.from_url("redis://localhost:6380")
index_8 = SearchIndex.from_dict(schema, redis_client=client_8, validate_on_load=True)

index_8.create(overwrite=True)
index_8.load(data)
print(f"{index_8.info()['num_docs']=}")

index_7_4.delete(drop=True)
index_8.delete(drop=True)
