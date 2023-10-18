import os
import pytest
import asyncio

from redisvl.utils.connection import (
    get_async_redis_connection,
    get_redis_connection
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

aredis = get_async_redis_connection(REDIS_URL)
redis = get_redis_connection(REDIS_URL)

@pytest.fixture()
def redis_url():
    return REDIS_URL

@pytest.fixture
def async_client():
    return aredis

@pytest.fixture
def client():
    return redis


@pytest.fixture
def openai_key():
    return os.getenv("OPENAI_API_KEY")

@pytest.fixture
def gcp_location():
    return os.getenv("GCP_LOCATION")

@pytest.fixture
def gcp_project_id():
    return os.getenv("GCP_PROJECT_ID")

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def clear_db():
    redis.flushall()
    yield
    redis.flushall()