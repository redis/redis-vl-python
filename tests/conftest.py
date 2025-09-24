import logging
import os
import subprocess
from datetime import datetime, timezone

import pytest
from testcontainers.compose import DockerCompose

from redisvl.index.index import AsyncSearchIndex, SearchIndex
from redisvl.redis.connection import RedisConnectionFactory, compare_versions
from redisvl.redis.utils import array_to_buffer
from redisvl.utils.vectorize import HFTextVectorizer

logger = logging.getLogger(__name__)


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
def redis_cluster_container(worker_id):
    project_name = f"redis_test_cluster_{worker_id}"
    # Use cwd if not running in GitHub Actions
    pwd = os.getcwd()
    compose_file = os.path.join(
        os.environ.get("GITHUB_WORKSPACE", pwd), "tests", "cluster-compose.yml"
    )
    os.environ["COMPOSE_PROJECT_NAME"] = (
        project_name  # For docker compose to pick it up if needed
    )
    # redis-stack-server comes up without modules in cluster mode, so we hard-code
    # the Redis 8 image for now.
    os.environ.setdefault("REDIS_IMAGE", "redis:8")

    # The DockerCompose helper isn't working with multiple services because the
    # subprocess command returns non-zero exit codes even on successful
    # completion. Here, we run the commands manually.

    # First attempt the docker-compose up command and handle its errors directly
    docker_cmd = [
        "docker",
        "compose",
        "-f",
        compose_file,
        "-p",  # Explicitly pass project name
        project_name,
        "up",
        "--wait",  # Wait for healthchecks
        "-d",  # Detach
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            check=False,  # Don't raise exception, we'll handle it ourselves
        )

        if result.returncode != 0:
            logger.error(f"Docker Compose up failed with exit code {result.returncode}")
            if result.stdout:
                logger.error(
                    f"STDOUT: {result.stdout.decode('utf-8', errors='replace')}"
                )
            if result.stderr:
                logger.error(
                    f"STDERR: {result.stderr.decode('utf-8', errors='replace')}"
                )

            # Try to get logs for more details
            logger.info("Attempting to fetch container logs...")
            try:
                logs_result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        compose_file,
                        "-p",
                        project_name,
                        "logs",
                    ],
                    capture_output=True,
                    text=True,
                )
                logger.info("Docker Compose logs:\n%s", logs_result.stdout)
                if logs_result.stderr:
                    logger.error("Docker Compose logs stderr: \n%s", logs_result.stderr)
            except Exception as log_e:
                logger.error(f"Failed to get Docker Compose logs: {repr(log_e)}")

            # Now raise the exception with the original result
            raise subprocess.CalledProcessError(
                result.returncode,
                docker_cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        # If we get here, setup was successful
        yield
    finally:
        # Always clean up
        try:
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    compose_file,
                    "-p",
                    project_name,
                    "down",
                    "-v",  # Remove volumes
                ],
                check=False,  # Don't raise on cleanup failure
                capture_output=True,
            )
        except Exception as e:
            logger.error(f"Error during cleanup: {repr(e)}")


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """
    Use the `DockerCompose` fixture to get host/port of the 'redis' service
    on container port 6379 (mapped to an ephemeral port on the host).
    """
    host, port = redis_container.get_service_host_and_port("redis", 6379)
    return f"redis://{host}:{port}"


@pytest.fixture(scope="session")
def redis_cluster_url(redis_cluster_container):
    # Hard-coded due to Docker issues
    return "redis://localhost:7001"


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


@pytest.fixture
def cluster_client(redis_cluster_url):
    """
    A sync Redis client that uses the dynamic `redis_cluster_url`.
    """
    conn = RedisConnectionFactory.get_redis_cluster_connection(
        redis_url=redis_cluster_url
    )
    yield conn


@pytest.fixture(scope="session")
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
    parser.addoption(
        "--run-cluster-tests",
        action="store_true",
        default=False,
        help="Run tests that require a Redis cluster",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys"
    )
    config.addinivalue_line(
        "markers", "requires_cluster: mark test as requiring a Redis cluster"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # Check each flag independently
    run_api_tests = config.getoption("--run-api-tests")
    run_cluster_tests = config.getoption("--run-cluster-tests")

    # Create skip markers
    skip_api = pytest.mark.skip(
        reason="Skipping test because API keys are not provided. Use --run-api-tests to run these tests."
    )
    skip_cluster = pytest.mark.skip(
        reason="Skipping test because Redis cluster is not available. Use --run-cluster-tests to run these tests."
    )

    # Apply skip markers independently based on flags
    for item in items:
        if item.get_closest_marker("requires_api_keys") and not run_api_tests:
            item.add_marker(skip_api)
        if item.get_closest_marker("requires_cluster") and not run_cluster_tests:
            item.add_marker(skip_cluster)


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


def has_redisearch_module(client):
    """Check if RediSearch module is available."""
    try:
        # Try to list indices - this is a RediSearch command
        client.execute_command("FT._LIST")
        return True
    except Exception:
        return False


async def has_redisearch_module_async(client):
    """Check if RediSearch module is available (async)."""
    try:
        # Try to list indices - this is a RediSearch command
        await client.execute_command("FT._LIST")
        return True
    except Exception:
        return False


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


def skip_if_no_redisearch(client, message: str = None):
    """
    Skip test if RediSearch module is not available.

    Args:
        client: Redis client instance
        message: Custom skip message
    """
    if not has_redisearch_module(client):
        skip_msg = message or "RediSearch module not available"
        pytest.skip(skip_msg)


async def skip_if_no_redisearch_async(client, message: str = None):
    """
    Skip test if RediSearch module is not available (async version).

    Args:
        client: Async Redis client instance
        message: Custom skip message
    """
    if not await has_redisearch_module_async(client):
        skip_msg = message or "RediSearch module not available"
        pytest.skip(skip_msg)
