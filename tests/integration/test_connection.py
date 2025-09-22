import os

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ConnectionError

from redisvl.redis.connection import (
    RedisConnectionFactory,
    convert_index_info_to_schema,
    unpack_redis_modules,
)
from redisvl.schema import IndexSchema
from redisvl.version import __version__
from tests.conftest import (
    skip_if_redis_version_below,
    skip_if_redis_version_below_async,
)

EXPECTED_LIB_NAME = f"redis-py(redisvl_v{__version__})"

# Remove after we remove connect() method from RedisConnectionFactory
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_unpack_redis_modules():
    module_list = [
        {
            "name": "search",
            "ver": 20811,
            "path": "/opt/redis-stack/lib/redisearch.so",
            "args": [],
        },
        {
            "name": "ReJSON",
            "ver": 20609,
            "path": "/opt/redis-stack/lib/rejson.so",
            "args": [],
        },
    ]
    installed_modules = unpack_redis_modules(module_list)
    assert installed_modules == {"search": 20811, "ReJSON": 20609}


def test_convert_index_info_to_schema():
    index_info = {
        "index_name": "image_summaries",
        "index_options": [],
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["summary"],
            "default_score",
            "1",
        ],
        "attributes": [
            [
                "identifier",
                "content",
                "attribute",
                "content",
                "type",
                "TEXT",
                "WEIGHT",
                "1",
            ],
            [
                "identifier",
                "doc_id",
                "attribute",
                "doc_id",
                "type",
                "TAG",
                "SEPARATOR",
                ",",
            ],
            [
                "identifier",
                "content_vector",
                "attribute",
                "content_vector",
                "type",
                "VECTOR",
                "algorithm",
                "FLAT",
                "data_type",
                "FLOAT32",
                "dim",
                1536,
                "distance_metric",
                "COSINE",
            ],
        ],
    }
    schema_dict = convert_index_info_to_schema(index_info)
    assert "index" in schema_dict
    assert "fields" in schema_dict
    assert len(schema_dict["fields"]) == len(index_info["attributes"])

    schema = IndexSchema.from_dict(schema_dict)
    assert schema.index.name == index_info["index_name"]


class TestConnect:
    def test_sync_redis_connect(self, redis_url):
        client = RedisConnectionFactory.connect(redis_url)
        assert client is not None
        assert isinstance(client, Redis)
        # Perform a simple operation
        assert client.ping()

    @pytest.mark.asyncio
    async def test_async_redis_connect(self, redis_url):
        client = RedisConnectionFactory.connect(redis_url, use_async=True)
        assert client is not None
        assert isinstance(client, AsyncRedis)
        # Perform a simple operation
        assert await client.ping()

    def test_missing_env_var(self):
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            del os.environ["REDIS_URL"]
            with pytest.raises(ValueError):
                RedisConnectionFactory.connect()
            os.environ["REDIS_URL"] = redis_url

    def test_invalid_url_format(self):
        with pytest.raises(ValueError):
            RedisConnectionFactory.connect(redis_url="invalid_url_format")

    def test_unknown_redis(self):
        with pytest.raises(ConnectionError):
            bad_client = RedisConnectionFactory.connect(redis_url="redis://fake:1234")
            bad_client.ping()


def test_validate_redis(client):
    skip_if_redis_version_below(client, "7.2.0")
    RedisConnectionFactory.validate_sync_redis(client)
    lib_name = client.client_info()
    assert lib_name["lib-name"] == EXPECTED_LIB_NAME


@pytest.mark.asyncio
async def test_validate_async_redis(async_client):
    await skip_if_redis_version_below_async(async_client, "7.2.0")
    await RedisConnectionFactory.validate_async_redis(async_client)
    lib_name = await async_client.client_info()
    assert lib_name["lib-name"] == EXPECTED_LIB_NAME


def test_validate_redis_custom_lib_name(client):
    skip_if_redis_version_below(client, "7.2.0")
    RedisConnectionFactory.validate_sync_redis(client, "langchain_v0.1.0")
    lib_name = client.client_info()
    assert lib_name["lib-name"] == f"redis-py(redisvl_v{__version__};langchain_v0.1.0)"


@pytest.mark.asyncio
async def test_validate_async_redis_custom_lib_name(async_client):
    await skip_if_redis_version_below_async(async_client, "7.2.0")
    await RedisConnectionFactory.validate_async_redis(async_client, "langchain_v0.1.0")
    lib_name = await async_client.client_info()
    assert lib_name["lib-name"] == f"redis-py(redisvl_v{__version__};langchain_v0.1.0)"
