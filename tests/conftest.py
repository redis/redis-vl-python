import os
from datetime import datetime, timezone

import pytest
from testcontainers.compose import DockerCompose

from redisvl.exceptions import RedisModuleVersionError
from redisvl.index.index import AsyncSearchIndex, SearchIndex
from redisvl.redis.connection import RedisConnectionFactory, compare_versions
from redisvl.redis.utils import array_to_buffer
from redisvl.utils.vectorize import HFTextVectorizer


@pytest.fixture(scope="session")
def worker_id(request):
    """
    Get the worker ID for the current test.

    In pytest-xdist, the config has "workerid" in workerinput.
    This fixture abstracts that logic to provide a consistent worker_id
    across all tests.
    """
    workerinput = getattr(request.config, "workerinput", {})
    return workerinput.get("workerid", "master")


@pytest.fixture(autouse=True)
def set_tokenizers_parallelism():
    """Disable tokenizers parallelism in tests to avoid deadlocks"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session", autouse=True)
def redis_container(worker_id):
    """
    If using xdist, create a unique Compose project for each xdist worker by
    setting COMPOSE_PROJECT_NAME. That prevents collisions on container/volume
    names.
    """
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


@pytest.fixture(scope="session")
def hf_vectorizer_float16():
    return HFTextVectorizer(dtype="float16")


@pytest.fixture(scope="session")
def hf_vectorizer_with_model():
    return HFTextVectorizer("sentence-transformers/all-mpnet-base-v2")


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
            "description": "engineers conduct trains that ride on train tracks",
            "last_updated": sample_datetimes["low"].timestamp(),
            "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
        },
        {
            "user": "mary",
            "age": 14,
            "job": "doctor",
            "description": "a medical professional who treats diseases and helps people stay healthy",
            "last_updated": sample_datetimes["low"].timestamp(),
            "credit_score": "low",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
        },
        {
            "user": "nancy",
            "age": 94,
            "job": "doctor",
            "description": "a research scientist specializing in cancers and diseases of the lungs",
            "last_updated": sample_datetimes["mid"].timestamp(),
            "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.7, 0.1, 0.5],
        },
        {
            "user": "tyler",
            "age": 100,
            "job": "engineer",
            "description": "a software developer with expertise in mathematics and computer science",
            "last_updated": sample_datetimes["mid"].timestamp(),
            "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.1, 0.4, 0.5],
        },
        {
            "user": "tim",
            "age": 12,
            "job": "dermatologist",
            "description": "a medical professional specializing in diseases of the skin",
            "last_updated": sample_datetimes["mid"].timestamp(),
            "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.4, 0.4, 0.5],
        },
        {
            "user": "taimur",
            "age": 15,
            "job": "CEO",
            "description": "high stress, but financially rewarding position at the head of a company",
            "last_updated": sample_datetimes["high"].timestamp(),
            "credit_score": "low",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.6, 0.1, 0.5],
        },
        {
            "user": "joe",
            "age": 35,
            "job": "dentist",
            "description": "like the tooth fairy because they'll take your teeth, but you have to pay them!",
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


@pytest.fixture
def flat_index(sample_data, redis_url, worker_id):
    """
    A fixture that uses the "flag" algorithm for its vector field.
    """

    # construct a search index from the schema
    index = SearchIndex.from_dict(
        {
            "index": {
                "name": f"user_index_{worker_id}",
                "prefix": f"v1_{worker_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "description", "type": "text"},
                {"name": "credit_score", "type": "tag"},
                {"name": "job", "type": "text"},
                {"name": "age", "type": "numeric"},
                {"name": "last_updated", "type": "numeric"},
                {"name": "location", "type": "geo"},
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
        },
        redis_url=redis_url,
    )

    # create the index (no data yet)
    index.create(overwrite=True)

    # Prepare and load the data
    def hash_preprocess(item: dict) -> dict:
        return {
            **item,
            "user_embedding": array_to_buffer(item["user_embedding"], "float32"),
        }

    index.load(sample_data, preprocess=hash_preprocess)

    # run the test
    yield index

    # clean up
    index.delete(drop=True)


@pytest.fixture
async def async_flat_index(sample_data, redis_url, worker_id):
    """
    A fixture that uses the "flag" algorithm for its vector field.
    """

    # construct a search index from the schema
    index = AsyncSearchIndex.from_dict(
        {
            "index": {
                "name": f"user_index_{worker_id}",
                "prefix": f"v1_{worker_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "description", "type": "text"},
                {"name": "credit_score", "type": "tag"},
                {"name": "job", "type": "text"},
                {"name": "age", "type": "numeric"},
                {"name": "last_updated", "type": "numeric"},
                {"name": "location", "type": "geo"},
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
        },
        redis_url=redis_url,
    )

    # create the index (no data yet)
    await index.create(overwrite=True)

    # Prepare and load the data
    def hash_preprocess(item: dict) -> dict:
        return {
            **item,
            "user_embedding": array_to_buffer(item["user_embedding"], "float32"),
        }

    await index.load(sample_data, preprocess=hash_preprocess)

    # run the test
    yield index

    # clean up
    await index.delete(drop=True)


@pytest.fixture
async def async_hnsw_index(sample_data, redis_url, worker_id):
    """
    A fixture that uses the "hnsw" algorithm for its vector field.
    """

    index = AsyncSearchIndex.from_dict(
        {
            "index": {
                "name": f"user_index_{worker_id}",
                "prefix": f"v1_{worker_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "description", "type": "text"},
                {"name": "credit_score", "type": "tag"},
                {"name": "job", "type": "text"},
                {"name": "age", "type": "numeric"},
                {"name": "last_updated", "type": "numeric"},
                {"name": "location", "type": "geo"},
                {
                    "name": "user_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 3,
                        "distance_metric": "cosine",
                        "algorithm": "hnsw",
                        "datatype": "float32",
                    },
                },
            ],
        },
        redis_url=redis_url,
    )

    # create the index (no data yet)
    await index.create(overwrite=True)

    # Prepare and load the data
    def hash_preprocess(item: dict) -> dict:
        return {
            **item,
            "user_embedding": array_to_buffer(item["user_embedding"], "float32"),
        }

    await index.load(sample_data, preprocess=hash_preprocess)

    # run the test
    yield index


@pytest.fixture
def hnsw_index(sample_data, redis_url, worker_id):
    """
    A fixture that uses the "hnsw" algorithm for its vector field.
    """

    index = SearchIndex.from_dict(
        {
            "index": {
                "name": f"user_index_{worker_id}",
                "prefix": f"v1_{worker_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "description", "type": "text"},
                {"name": "credit_score", "type": "tag"},
                {"name": "job", "type": "text"},
                {"name": "age", "type": "numeric"},
                {"name": "last_updated", "type": "numeric"},
                {"name": "location", "type": "geo"},
                {
                    "name": "user_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 3,
                        "distance_metric": "cosine",
                        "algorithm": "hnsw",
                        "datatype": "float32",
                    },
                },
            ],
        },
        redis_url=redis_url,
    )

    # create the index (no data yet)
    index.create(overwrite=True)

    # Prepare and load the data
    def hash_preprocess(item: dict) -> dict:
        return {
            **item,
            "user_embedding": array_to_buffer(item["user_embedding"], "float32"),
        }

    index.load(sample_data, preprocess=hash_preprocess)

    # run the test
    yield index


# Version checking utilities
def get_redis_version(client):
    """Get Redis version from client info."""
    return client.info()["redis_version"]


async def get_redis_version_async(client):
    """Get Redis version from async client info."""
    info = await client.info()
    return info["redis_version"]


def skip_if_redis_version_below(client, min_version: str, message: str = None):
    """
    Skip test if Redis version is below minimum required.

    Args:
        client: Redis client instance
        min_version: Minimum required Redis version
        message: Custom skip message
    """
    redis_version = get_redis_version(client)
    if not compare_versions(redis_version, min_version):
        skip_msg = message or f"Redis version {redis_version} < {min_version} required"
        pytest.skip(skip_msg)


async def skip_if_redis_version_below_async(
    client, min_version: str, message: str = None
):
    """
    Skip test if Redis version is below minimum required (async version).

    Args:
        client: Async Redis client instance
        min_version: Minimum required Redis version
        message: Custom skip message
    """
    redis_version = await get_redis_version_async(client)
    if not compare_versions(redis_version, min_version):
        skip_msg = message or f"Redis version {redis_version} < {min_version} required"
        pytest.skip(skip_msg)


def skip_if_module_version_error(func, *args, **kwargs):
    """
    Execute function and skip test if RedisModuleVersionError is raised.

    Args:
        func: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
    """
    try:
        return func(*args, **kwargs)
    except RedisModuleVersionError:
        pytest.skip("Required Redis modules not available or version too low")


async def skip_if_module_version_error_async(func, *args, **kwargs):
    """
    Execute async function and skip test if RedisModuleVersionError is raised.

    Args:
        func: Async function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
    """
    try:
        return await func(*args, **kwargs)
    except RedisModuleVersionError:
        pytest.skip("Required Redis modules not available or version too low")
