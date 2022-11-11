import os
import pytest

from redisvl.utils.connection import get_async_redis_connection

HOST = os.environ.get("REDIS_HOST", "localhost")
PORT = os.environ.get("REDIS_PORT", 6379)
USER = os.environ.get("REDIS_USER", "default")
PASS = os.environ.get("REDIS_PASSWORD", "")

@pytest.fixture
def async_redis():
    return get_async_redis_connection(HOST, PORT, PASS)