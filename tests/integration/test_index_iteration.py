import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query.filter import Tag

DOCS = [
    {"id": "1", "category": "A"},
    {"id": "2", "category": "B"},
    {"id": "3", "category": "A"},
    {"id": "4", "category": "C"},
]


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
    # id_field makes the key deterministic: <prefix>:<id>
    index.load(DOCS, id_field="id")
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
    await index.load(DOCS, id_field="id")
    yield index
    await index.delete(drop=True)


def test_iter_yields_every_key(sample_index):
    """iter() with no filter must yield every key in the index, once each."""
    keys = list(sample_index.iter())

    assert len(keys) == 4
    assert set(keys) == {f"{sample_index.prefix}:{i}" for i in range(1, 5)}


def test_iter_respects_filter_expression(sample_index):
    """A filter expression must narrow the yielded keys."""
    keys = list(sample_index.iter(filter_expression=Tag("category") == "A"))

    assert set(keys) == {f"{sample_index.prefix}:1", f"{sample_index.prefix}:3"}


def test_iter_is_lazy(sample_index):
    """Iteration must stream: the first key arrives without draining the index."""
    iterator = sample_index.iter()

    assert next(iterator) is not None


def test_iter_pages_when_batch_size_is_smaller_than_the_index(sample_index):
    """A batch_size below the document count must still yield every key exactly once."""
    keys = list(sample_index.iter(batch_size=2))

    assert sorted(keys) == sorted(f"{sample_index.prefix}:{i}" for i in range(1, 5))


@pytest.mark.asyncio
async def test_aiter_yields_every_key(async_sample_index):
    """aiter() must mirror iter() on the async client."""
    keys = [key async for key in async_sample_index.aiter()]

    assert len(keys) == 4
    assert set(keys) == {f"{async_sample_index.prefix}:{i}" for i in range(1, 5)}


@pytest.mark.asyncio
async def test_aiter_respects_filter_expression(async_sample_index):
    """The async iterator must apply the filter the same way the sync one does."""
    keys = [
        key
        async for key in async_sample_index.aiter(
            filter_expression=Tag("category") == "A"
        )
    ]

    assert set(keys) == {
        f"{async_sample_index.prefix}:1",
        f"{async_sample_index.prefix}:3",
    }


@pytest.mark.asyncio
async def test_aiter_pages_when_batch_size_is_smaller_than_the_index(
    async_sample_index,
):
    """A batch_size below the document count must still yield every key exactly once."""
    keys = [key async for key in async_sample_index.aiter(batch_size=2)]

    assert sorted(keys) == sorted(
        f"{async_sample_index.prefix}:{i}" for i in range(1, 5)
    )
