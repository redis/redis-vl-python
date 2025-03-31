import os
from datetime import datetime, timezone

import pytest
from testcontainers.compose import DockerCompose

from redisvl.redis.connection import RedisConnectionFactory
from redisvl.utils.vectorize import HFTextVectorizer


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
    os.environ.setdefault("REDIS_IMAGE", "redis/redis-stack-server:latest")

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
    async with await RedisConnectionFactory._get_aredis_connection(redis_url) as client:
        yield client


@pytest.fixture
def client(redis_url):
    """
    A sync Redis client that uses the dynamic `redis_url`.
    """
    conn = RedisConnectionFactory.get_redis_connection(redis_url=redis_url)
    yield conn


@pytest.fixture(scope="session", autouse=True)
def hf_vectorizer():
    return HFTextVectorizer(
        model="sentence-transformers/all-mpnet-base-v2",
        token=os.getenv("HF_TOKEN"),
        cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME"),
    )


@pytest.fixture
def sample_datetimes():
    return {
        "low": datetime(2025, 1, 16, 13).astimezone(timezone.utc),
        "mid": datetime(2025, 2, 16, 13).astimezone(timezone.utc),
        "high": datetime(2025, 3, 16, 13).astimezone(timezone.utc),
    }


@pytest.fixture
def sample_data(sample_datetimes):
    return [
        {
            "user": "john",
            "age": 18,
            "job": "engineer",
            "last_updated": sample_datetimes["low"].timestamp(),
            "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
        },
        {
            "user": "mary",
            "age": 14,
            "job": "doctor",
            "last_updated": sample_datetimes["low"].timestamp(),
            "credit_score": "low",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
        },
        {
            "user": "nancy",
            "age": 94,
            "job": "doctor",
            "last_updated": sample_datetimes["mid"].timestamp(),
            "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.7, 0.1, 0.5],
        },
        {
            "user": "tyler",
            "age": 100,
            "job": "engineer",
            "last_updated": sample_datetimes["mid"].timestamp(),
            "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.1, 0.4, 0.5],
        },
        {
            "user": "tim",
            "age": 12,
            "job": "dermatologist",
            "last_updated": sample_datetimes["mid"].timestamp(),
            "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.4, 0.4, 0.5],
        },
        {
            "user": "taimur",
            "age": 15,
            "job": "CEO",
            "last_updated": sample_datetimes["high"].timestamp(),
            "credit_score": "low",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.6, 0.1, 0.5],
        },
        {
            "user": "joe",
            "age": 35,
            "job": "dentist",
            "last_updated": sample_datetimes["high"].timestamp(),
            "credit_score": "medium",
            "location": "-110.0839,37.3861",
            "user_embedding": [-0.1, -0.1, -0.5],
        },
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API keys",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-api-tests"):
        return

    # Otherwise skip all tests requiring an API key
    skip_api = pytest.mark.skip(
        reason="Skipping test because API keys are not provided. Use --run-api-tests to run these tests."
    )
    for item in items:
        if item.get_closest_marker("requires_api_keys"):
            item.add_marker(skip_api)
