import os
import pytest
import asyncio

from redisvl.utils.connection import (
    get_async_redis_connection,
    get_redis_connection
)
REDIS_ADDRESS = os.getenv("REDIS_ADDRESS", "redis://localhost:6379")

aredis = get_async_redis_connection(REDIS_ADDRESS)
redis = get_redis_connection(REDIS_ADDRESS)


@pytest.fixture
def async_client():
    return aredis

@pytest.fixture
def client():
    return redis


@pytest.fixture
def openai_key():
    return os.getenv("OPENAI_KEY")


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()