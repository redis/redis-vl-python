import pytest
from redisvl.index import SearchIndex, AsyncSearchIndex
from redisvl.query.filter import Tag


@pytest.fixture
def sample_index(redis_url, redis_test_name):
    index_name = redis_test_name("iter_index")
    prefix = redis_test_name("iter_doc")
    index = SearchIndex.from_dict(
        {
            "index": {"name": index_name, "prefix": prefix, "storage_type": "hash"},
            "fields": [{"name": "category", "type": "tag"}],
        },
        redis_url=redis_url,
    )
    index.create(overwrite=True)
    docs = [
        {"id": f"{prefix}:1", "category": "A"},
        {"id": f"{prefix}:2", "category": "B"},
        {"id": f"{prefix}:3", "category": "A"},
        {"id": f"{prefix}:4", "category": "C"},
    ]
    index.load(docs)
    yield index
    index.delete(drop=True)


@pytest.fixture
async def async_sample_index(redis_url, redis_test_name):
    index_name = redis_test_name("async_iter_index")
    prefix = redis_test_name("async_iter_doc")
    index = AsyncSearchIndex.from_dict(
        {
            "index": {"name": index_name, "prefix": prefix, "storage_type": "hash"},
            "fields": [{"name": "category", "type": "tag"}],
        },
        redis_url=redis_url,
    )
    await index.create(overwrite=True)
    docs = [
        {"id": f"{prefix}:1", "category": "A"},
        {"id": f"{prefix}:2", "category": "B"},
        {"id": f"{prefix}:3", "category": "A"},
        {"id": f"{prefix}:4", "category": "C"},
    ]
    await index.load(docs)
    yield index
    await index.delete(drop=True)


def test_sync_index_iter(sample_index):
    # Iterate all keys
    all_keys = list(sample_index.iter())
    assert len(all_keys) == 4
    assert set(all_keys) == {
        f"{sample_index.prefix}:1",
        f"{sample_index.prefix}:2",
        f"{sample_index.prefix}:3",
        f"{sample_index.prefix}:4",
    }

    # Iterate with filter
    filter_a = Tag("category") == "A"
    filtered_keys = list(sample_index.iter(filter_expression=filter_a))
    assert len(filtered_keys) == 2
    assert set(filtered_keys) == {
        f"{sample_index.prefix}:1",
        f"{sample_index.prefix}:3",
    }

    # Magic __iter__
    magic_keys = list(sample_index)
    assert len(magic_keys) == 4


@pytest.mark.asyncio
async def test_async_index_aiter(async_sample_index):
    # Iterate all keys asynchronously
    all_keys = []
    async for key in async_sample_index.aiter():
        all_keys.append(key)
    assert len(all_keys) == 4
    assert set(all_keys) == {
        f"{async_sample_index.prefix}:1",
        f"{async_sample_index.prefix}:2",
        f"{async_sample_index.prefix}:3",
        f"{async_sample_index.prefix}:4",
    }

    # Iterate with filter asynchronously
    filter_a = Tag("category") == "A"
    filtered_keys = []
    async for key in async_sample_index.aiter(filter_expression=filter_a):
        filtered_keys.append(key)
    assert len(filtered_keys) == 2
    assert set(filtered_keys) == {
        f"{async_sample_index.prefix}:1",
        f"{async_sample_index.prefix}:3",
    }

    # Magic __aiter__
    magic_keys = []
    async for key in async_sample_index:
        magic_keys.append(key)
    assert len(magic_keys) == 4
