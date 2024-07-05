import os
import pytest
import asyncio

from redisvl.redis.connection import RedisConnectionFactory
from testcontainers.compose import DockerCompose



@pytest.fixture(scope="session", autouse=True)
def redis_container():
    # Set the default Redis version if not already set
    os.environ.setdefault("REDIS_VERSION", "edge")

    compose = DockerCompose("tests", compose_file_name="docker-compose.yml", pull=True)
    compose.start()

    redis_host, redis_port = compose.get_service_host_and_port("redis", 6379)
    redis_url = f"redis://{redis_host}:{redis_port}"
    os.environ["REDIS_URL"] = redis_url

    yield compose

    compose.stop()

@pytest.fixture(scope="session")
def redis_url():
    return os.getenv("REDIS_URL", "redis://localhost:6379")

@pytest.fixture
async def async_client(redis_url):
    client = await RedisConnectionFactory.get_async_redis_connection(redis_url)
    yield client
    try:
        await client.aclose()
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise

@pytest.fixture
def client():
    conn = RedisConnectionFactory.get_redis_connection(os.environ["REDIS_URL"])
    yield conn
    conn.close()

@pytest.fixture
def openai_key():
    return os.getenv("OPENAI_API_KEY")

@pytest.fixture
def openai_version():
    return os.getenv("OPENAI_API_VERSION")

@pytest.fixture
def azure_endpoint():
    return os.getenv("AZURE_OPENAI_ENDPOINT")

@pytest.fixture
def cohere_key():
    return os.getenv("COHERE_API_KEY")

@pytest.fixture
def mistral_key():
    return os.getenv("MISTRAL_API_KEY")

@pytest.fixture
def gcp_location():
    return os.getenv("GCP_LOCATION")

@pytest.fixture
def gcp_project_id():
    return os.getenv("GCP_PROJECT_ID")


@pytest.fixture
def sample_data():
    return [
    {
        "user": "john",
        "age": 18,
        "job": "engineer",
        "credit_score": "high",
        "location": "-122.4194,37.7749",
        "user_embedding": [0.1, 0.1, 0.5]
    },
    {
        "user": "mary",
        "age": 14,
        "job": "doctor",
        "credit_score": "low",
        "location": "-122.4194,37.7749",
        "user_embedding": [0.1, 0.1, 0.5]
    },
    {
        "user": "nancy",
        "age": 94,
        "job": "doctor",
        "credit_score": "high",
        "location": "-122.4194,37.7749",
        "user_embedding": [0.7, 0.1, 0.5]
    },
    {
        "user": "tyler",
        "age": 100,
        "job": "engineer",
        "credit_score": "high",
        "location": "-110.0839,37.3861",
        "user_embedding": [0.1, 0.4, 0.5]
    },
    {
        "user": "tim",
        "age": 12,
        "job": "dermatologist",
        "credit_score": "high",
        "location": "-110.0839,37.3861",
        "user_embedding": [0.4, 0.4, 0.5]
    },
    {
        "user": "taimur",
        "age": 15,
        "job": "CEO",
        "credit_score": "low",
        "location": "-110.0839,37.3861",
        "user_embedding": [0.6, 0.1, 0.5]
    },
    {
        "user": "joe",
        "age": 35,
        "job": "dentist",
        "credit_score": "medium",
        "location": "-110.0839,37.3861",
        "user_embedding": [0.9, 0.9, 0.1]
    },
]

@pytest.fixture
def clear_db(redis):
    redis.flushall()
    yield
    redis.flushall()

@pytest.fixture
def app_name():
    return "test_app"

@pytest.fixture
def session_tag():
    return "123"

@pytest.fixture
def user_tag():
    return "abc"
