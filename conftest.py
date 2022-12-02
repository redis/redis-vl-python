import os
import pytest
import pandas as pd

from redisvl.utils.connection import (
    get_async_redis_connection,
    get_redis_connection
)

HOST = os.environ.get("REDIS_HOST", "localhost")
PORT = os.environ.get("REDIS_PORT", 6379)
USER = os.environ.get("REDIS_USER", "default")
PASS = os.environ.get("REDIS_PASSWORD", "")

aredis = get_async_redis_connection(HOST, PORT, PASS)
redis = get_redis_connection(HOST, PORT, PASS)
print(type(redis))

@pytest.fixture
def async_client():
    return aredis

@pytest.fixture
def client():
    return redis

@pytest.fixture
def df():

    data = pd.DataFrame(
        {
            "users": ["john", "mary", "joe"],
            "age": [1, 2, 3],
            "job": ["engineer", "doctor", "dentist"],
            "credit_score": ["high", "low", "medium"],
            "user_embedding": [
                [0.1, 0.1, 0.5],
                [0.1, 0.1, 0.5],
                [0.9, 0.9, 0.1],
            ],
        }
    )
    return data
