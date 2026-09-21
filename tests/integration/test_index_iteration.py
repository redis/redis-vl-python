import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query.filter import Tag
from redisvl.utils.utils import contextlib

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


def test_iter_keys_yields_every_key(sample_index):
    keys = list(sample_index.iter_keys())

    assert len(keys) == 4
    assert set(keys) == {f"{sample_index.prefix}:{i}" for i in range(1, 5)}


def test_iter_keys_respects_filter_expression(sample_index):
    keys = list(sample_index.iter_keys(filter_expression=Tag("category") == "A"))

    assert set(keys) == {f"{sample_index.prefix}:1", f"{sample_index.prefix}:3"}


def test_iter_keys_is_lazy(sample_index):
    iterator = sample_index.iter_keys()

    assert next(iterator) is not None


def test_iter_keys_pages_when_batch_size_is_smaller_than_the_index(sample_index):
    keys = list(sample_index.iter_keys(batch_size=2))

    assert sorted(keys) == sorted(f"{sample_index.prefix}:{i}" for i in range(1, 5))


@pytest.mark.asyncio
async def test_aiter_keys_yields_every_key(async_sample_index):
    keys = [key async for key in async_sample_index.aiter_keys()]

    assert len(keys) == 4
    assert set(keys) == {f"{async_sample_index.prefix}:{i}" for i in range(1, 5)}


@pytest.mark.asyncio
async def test_aiter_keys_respects_filter_expression(async_sample_index):
    keys = [
        key
        async for key in async_sample_index.aiter_keys(
            filter_expression=Tag("category") == "A"
        )
    ]

    assert set(keys) == {
        f"{async_sample_index.prefix}:1",
        f"{async_sample_index.prefix}:3",
    }


def test_iter_keys_raises_with_non_int_batch_size(sample_index):
    with pytest.raises(TypeError):
        list(sample_index.iter_keys(batch_size="5"))


def test_iter_keys_raises_with_zero_batch_size(sample_index):
    with pytest.raises(ValueError):
        list(sample_index.iter_keys(batch_size=0))


def test_iter_keys_raises_with_negative_batch_size(sample_index):
    with pytest.raises(ValueError):
        list(sample_index.iter_keys(batch_size=-1))
