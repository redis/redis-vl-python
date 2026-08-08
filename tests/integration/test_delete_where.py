import pytest
from redisvl.index import SearchIndex, AsyncSearchIndex
from redisvl.query.filter import Tag


@pytest.fixture
def sample_delete_index(redis_url, redis_test_name):
    index_name = redis_test_name("delete_where_index")
    prefix = redis_test_name("delete_where_doc")
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
async def async_sample_delete_index(redis_url, redis_test_name):
    index_name = redis_test_name("async_delete_where_index")
    prefix = redis_test_name("async_delete_where_doc")
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


def test_sync_delete_where(sample_delete_index):
    # Delete category A
    deleted = sample_delete_index.delete_where(Tag("category") == "A")
    assert deleted == 2

    remaining = list(sample_delete_index.iter())
    assert len(remaining) == 2
    assert set(remaining) == {
        f"{sample_delete_index.prefix}:2",
        f"{sample_delete_index.prefix}:4",
    }

    # Clear remaining
    cleared = sample_delete_index.clear()
    assert cleared == 2
    assert len(list(sample_delete_index.iter())) == 0


@pytest.mark.asyncio
async def test_async_adelete_where(async_sample_delete_index):
    # Delete category A asynchronously
    deleted = await async_sample_delete_index.adelete_where(Tag("category") == "A")
    assert deleted == 2

    remaining = []
    async for key in async_sample_delete_index.aiter():
        remaining.append(key)
    assert len(remaining) == 2
    assert set(remaining) == {
        f"{async_sample_delete_index.prefix}:2",
        f"{async_sample_delete_index.prefix}:4",
    }

    # Clear remaining asynchronously
    cleared = await async_sample_delete_index.clear()
    assert cleared == 2
    remaining_after = []
    async for key in async_sample_delete_index.aiter():
        remaining_after.append(key)
    assert len(remaining_after) == 0
