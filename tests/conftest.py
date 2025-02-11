import os

import pytest
from testcontainers.compose import DockerCompose

from redisvl.redis.connection import RedisConnectionFactory


@pytest.fixture(autouse=True)
def set_tokenizers_parallelism():
    """Disable tokenizers parallelism in tests to avoid deadlocks"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session", autouse=True)
def redis_container(request):
    """
    If using xdist, create a unique Compose project for each xdist worker by
    setting COMPOSE_PROJECT_NAME. That prevents collisions on container/volume
    names.
    """
    # In xdist, the config has "workerid" in workerinput
    workerinput = getattr(request.config, "workerinput", {})
    worker_id = workerinput.get("workerid", "master")

    # Set the Compose project name so containers do not clash across workers
    os.environ["COMPOSE_PROJECT_NAME"] = f"redis_test_{worker_id}"
    os.environ.setdefault("REDIS_VERSION", "edge")

    compose = DockerCompose(
        context="tests",
        compose_file_name="docker-compose.yml",
        pull=True,
    )
    compose.start()

    yield compose

    compose.stop()


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """
    Use the `DockerCompose` fixture to get host/port of the 'redis' service
    on container port 6379 (mapped to an ephemeral port on the host).
    """
    host, port = redis_container.get_service_host_and_port("redis", 6379)
    return f"redis://{host}:{port}"


@pytest.fixture
async def async_client(redis_url):
    """
    An async Redis client that uses the dynamic `redis_url`.
    """
    client = await RedisConnectionFactory.get_async_redis_connection(redis_url)
    yield client
    try:
        await client.aclose()
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise


@pytest.fixture
def client(redis_url):
    """
    A sync Redis client that uses the dynamic `redis_url`.
    """
    conn = RedisConnectionFactory.get_redis_connection(redis_url)
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
def aws_credentials():
    return {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),
    }


@pytest.fixture
def sample_data():
    return [
        {
            "user": "john",
            "age": 18,
            "job": "engineer",
            "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
        },
        {
            "user": "mary",
            "age": 14,
            "job": "doctor",
            "credit_score": "low",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
        },
        {
            "user": "nancy",
            "age": 94,
            "job": "doctor",
            "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.7, 0.1, 0.5],
        },
        {
            "user": "tyler",
            "age": 100,
            "job": "engineer",
            "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.1, 0.4, 0.5],
        },
        {
            "user": "tim",
            "age": 12,
            "job": "dermatologist",
            "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.4, 0.4, 0.5],
        },
        {
            "user": "taimur",
            "age": 15,
            "job": "CEO",
            "credit_score": "low",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.6, 0.1, 0.5],
        },
        {
            "user": "joe",
            "age": 35,
            "job": "dentist",
            "credit_score": "medium",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.9, 0.9, 0.1],
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
