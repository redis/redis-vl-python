import os

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ConnectionError

from redisvl.exceptions import RedisModuleVersionError
from redisvl.redis.connection import (
    RedisConnectionFactory,
    compare_versions,
    convert_index_info_to_schema,
    get_address_from_env,
    unpack_redis_modules,
    validate_modules,
)
from redisvl.schema import IndexSchema
from redisvl.version import __version__

EXPECTED_LIB_NAME = f"redis-py(redisvl_v{__version__})"


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


def test_validate_modules_exist_search():
    validate_modules(
        installed_modules={"search": 20811},
        required_modules=[
            {"name": "search", "ver": 20600},
            {"name": "searchlight", "ver": 20600},
        ],
    )


def test_validate_modules_exist_searchlight():
    validate_modules(
        installed_modules={"searchlight": 20819},
        required_modules=[
            {"name": "search", "ver": 20810},
            {"name": "searchlight", "ver": 20810},
        ],
    )


def test_validate_modules_not_exist():
    with pytest.raises(RedisModuleVersionError):
        validate_modules(
            installed_modules={"search": 20811},
            required_modules=[
                {"name": "ReJSON", "ver": 20600},
            ],
        )


def test_sync_redis_connect(redis_url):
    client = RedisConnectionFactory.connect(redis_url)
    assert client is not None
    assert isinstance(client, Redis)
    # Perform a simple operation
    assert client.ping()


@pytest.mark.asyncio
async def test_async_redis_connect(redis_url):
    client = RedisConnectionFactory.connect(redis_url, use_async=True)
    assert client is not None
    assert isinstance(client, AsyncRedis)
    # Perform a simple operation
    assert await client.ping()


def test_missing_env_var():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        del os.environ["REDIS_URL"]
        with pytest.raises(ValueError):
            RedisConnectionFactory.connect()
        os.environ["REDIS_URL"] = redis_url


def test_invalid_url_format():
    with pytest.raises(ValueError):
        RedisConnectionFactory.connect(redis_url="invalid_url_format")


def test_unknown_redis():
    bad_client = RedisConnectionFactory.connect(redis_url="redis://fake:1234")
    with pytest.raises(ConnectionError):
        bad_client.ping()


def test_validate_redis(client):
    redis_version = client.info()["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")
    RedisConnectionFactory.validate_sync_redis(client)
    lib_name = client.client_info()
    assert lib_name["lib-name"] == EXPECTED_LIB_NAME


@pytest.mark.asyncio
async def test_validate_async_redis(async_client):
    redis_version = (await async_client.info())["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")
    await RedisConnectionFactory.validate_async_redis(async_client)
    lib_name = await async_client.client_info()
    assert lib_name["lib-name"] == EXPECTED_LIB_NAME


def test_validate_redis_custom_lib_name(client):
    redis_version = client.info()["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")
    RedisConnectionFactory.validate_sync_redis(client, "langchain_v0.1.0")
    lib_name = client.client_info()
    assert lib_name["lib-name"] == f"redis-py(redisvl_v{__version__};langchain_v0.1.0)"


@pytest.mark.asyncio
async def test_validate_async_redis_custom_lib_name(async_client):
    redis_version = (await async_client.info())["redis_version"]
    if not compare_versions(redis_version, "7.2.0"):
        pytest.skip("Not using a late enough version of Redis")
    await RedisConnectionFactory.validate_async_redis(async_client, "langchain_v0.1.0")
    lib_name = await async_client.client_info()
    assert lib_name["lib-name"] == f"redis-py(redisvl_v{__version__};langchain_v0.1.0)"
