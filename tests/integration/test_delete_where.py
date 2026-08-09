import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import FilterExpression, Tag

DOCS = [
    {"id": "1", "category": "A"},
    {"id": "2", "category": "B"},
    {"id": "3", "category": "A"},
    {"id": "4", "category": "C"},
]


def _all_keys(index):
    """Return every key in the index, without relying on other unmerged APIs."""
    query = FilterQuery(FilterExpression("*"), return_fields=["id"], num_results=100)
    return {record["id"] for record in index.query(query)}


async def _all_keys_async(index):
    query = FilterQuery(FilterExpression("*"), return_fields=["id"], num_results=100)
    return {record["id"] for record in await index.query(query)}


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
    index.load(DOCS, id_field="id")
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
    await index.load(DOCS, id_field="id")
    yield index
    await index.delete(drop=True)


def test_delete_where_removes_only_matching_documents(sample_delete_index):
    """A filtered delete must remove matches and leave everything else in place."""
    deleted = sample_delete_index.delete_where(Tag("category") == "A")

    assert deleted == 2
    assert _all_keys(sample_delete_index) == {
        f"{sample_delete_index.prefix}:2",
        f"{sample_delete_index.prefix}:4",
    }


def test_delete_where_with_no_matches_is_a_noop(sample_delete_index):
    """A filter matching nothing must delete nothing and report zero."""
    before = _all_keys(sample_delete_index)

    deleted = sample_delete_index.delete_where(Tag("category") == "no_such_value")

    assert deleted == 0
    assert _all_keys(sample_delete_index) == before


def test_clear_still_removes_everything(sample_delete_index):
    """clear() must keep its contract after delegating to the shared internals."""
    cleared = sample_delete_index.clear()

    assert cleared == 4
    assert _all_keys(sample_delete_index) == set()


def test_delete_where_pages_below_batch_size(sample_delete_index):
    """A batch_size smaller than the match count must still delete every match."""
    deleted = sample_delete_index.delete_where(FilterExpression("*"), batch_size=2)

    assert deleted == 4
    assert _all_keys(sample_delete_index) == set()


@pytest.mark.asyncio
async def test_adelete_where_removes_only_matching_documents(
    async_sample_delete_index,
):
    """The async client must behave identically to the sync one."""
    deleted = await async_sample_delete_index.adelete_where(Tag("category") == "A")

    assert deleted == 2
    assert await _all_keys_async(async_sample_delete_index) == {
        f"{async_sample_delete_index.prefix}:2",
        f"{async_sample_delete_index.prefix}:4",
    }


@pytest.mark.asyncio
async def test_adelete_where_with_no_matches_is_a_noop(async_sample_delete_index):
    """An async filter matching nothing must delete nothing and report zero."""
    before = await _all_keys_async(async_sample_delete_index)

    deleted = await async_sample_delete_index.adelete_where(
        Tag("category") == "no_such_value"
    )

    assert deleted == 0
    assert await _all_keys_async(async_sample_delete_index) == before


@pytest.mark.asyncio
async def test_async_clear_still_removes_everything(async_sample_delete_index):
    """The async clear() must keep its contract after the refactor."""
    cleared = await async_sample_delete_index.clear()

    assert cleared == 4
    assert await _all_keys_async(async_sample_delete_index) == set()
