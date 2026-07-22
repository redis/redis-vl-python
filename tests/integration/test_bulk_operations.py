"""Integration tests for bulk delete/update by filter (RAAE-1326)."""

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import CountQuery
from redisvl.query.filter import Num, Tag
from redisvl.schema import IndexSchema

BULK_FIELDS = [
    {"name": "cat", "type": "tag"},
    {"name": "status", "type": "tag"},
    {"name": "n", "type": "numeric", "attrs": {"sortable": True}},
]


def _schema(name, storage_type):
    return IndexSchema.from_dict(
        {
            "index": {
                "name": name,
                "prefix": name,
                "storage_type": storage_type,
            },
            "fields": BULK_FIELDS,
        }
    )


def _hash_data(n=40):
    return [
        {
            "id": str(i),
            "cat": "a" if i % 2 else "b",
            "status": "draft",
            "n": i,
            "keep": "orig",
        }
        for i in range(n)
    ]


@pytest.fixture
def hash_index(worker_id, client):
    index = SearchIndex(
        schema=_schema(f"bulk_h_{worker_id}", "hash"), redis_client=client
    )
    index.create(overwrite=True, drop=True)
    yield index
    index.delete(drop=True)


@pytest.fixture
def json_index(worker_id, client):
    index = SearchIndex(
        schema=_schema(f"bulk_j_{worker_id}", "json"), redis_client=client
    )
    index.create(overwrite=True, drop=True)
    yield index
    index.delete(drop=True)


# --------------------------------------------------------------------------- #
# delete_by_filter
# --------------------------------------------------------------------------- #
def test_delete_by_filter_removes_only_matches(hash_index):
    hash_index.load(_hash_data(), id_field="id")
    deleted = hash_index.delete_by_filter(Tag("cat") == "a", batch_size=10)
    assert deleted == 20
    assert hash_index.query(CountQuery(Tag("cat") == "a")) == 0
    assert hash_index.query(CountQuery(Tag("cat") == "b")) == 20


def test_delete_by_filter_dry_run_does_not_mutate(hash_index):
    hash_index.load(_hash_data(), id_field="id")
    count = hash_index.delete_by_filter(Num("n") < 10, dry_run=True)
    assert count == 10
    # nothing actually removed
    assert hash_index.query(CountQuery(Num("n") < 10)) == 10


def test_delete_by_filter_reports_progress(hash_index):
    hash_index.load(_hash_data(), id_field="id")
    progress = []
    hash_index.delete_by_filter(
        Tag("cat") == "a", batch_size=5, on_progress=progress.append
    )
    assert progress  # invoked at least once
    assert progress == sorted(progress)  # monotonically increasing
    assert progress[-1] == 20


@pytest.mark.parametrize("bad_filter", [None, "*", ""])
def test_delete_by_filter_guards_match_all(hash_index, bad_filter):
    hash_index.load(_hash_data(), id_field="id")
    with pytest.raises(ValueError):
        hash_index.delete_by_filter(bad_filter)
    # index untouched
    assert hash_index.query(CountQuery(Num("n") >= 0)) == 40


def test_delete_by_filter_allow_all_override(hash_index):
    hash_index.load(_hash_data(), id_field="id")
    deleted = hash_index.delete_by_filter("*", allow_all=True, batch_size=10)
    assert deleted == 40
    assert hash_index.query(CountQuery(Num("n") >= 0)) == 0


# --------------------------------------------------------------------------- #
# update_by_filter
# --------------------------------------------------------------------------- #
def test_update_by_filter_hash_is_partial(hash_index):
    hash_index.load(_hash_data(), id_field="id")
    updated = hash_index.update_by_filter(
        Tag("cat") == "b", {"status": "published"}, batch_size=10
    )
    assert updated == 20
    doc = hash_index.fetch("0")  # id 0 -> cat "b"
    assert doc["status"] == "published"
    assert doc["keep"] == "orig"  # untouched field preserved
    assert doc["cat"] == "b"
    # non-matching docs unchanged
    other = hash_index.fetch("1")  # cat "a"
    assert other["status"] == "draft"


def test_update_by_filter_json_merges_partially(json_index):
    json_index.load(
        [{"id": "x", "cat": "a", "status": "draft", "n": 1, "obj": {"p": 1, "q": 2}}],
        id_field="id",
    )
    json_index.update_by_filter(
        Tag("cat") == "a", {"status": "done", "obj": {"q": 9, "r": 3}}
    )
    doc = json_index.fetch("x")
    assert doc["status"] == "done"
    # nested object merged recursively, not replaced
    assert doc["obj"] == {"p": 1, "q": 9, "r": 3}


def test_update_by_filter_spans_multiple_batches(hash_index):
    # more docs than batch_size to exercise the cursor iteration
    hash_index.load(_hash_data(120), id_field="id")
    progress = []
    updated = hash_index.update_by_filter(
        Tag("cat") == "a", {"status": "x"}, batch_size=25, on_progress=progress.append
    )
    assert updated == 60
    assert len(progress) >= 2  # spanned multiple batches
    assert progress[-1] == 60


def test_update_by_filter_dry_run_and_empty_values(hash_index):
    hash_index.load(_hash_data(), id_field="id")
    assert (
        hash_index.update_by_filter(Tag("cat") == "a", {"status": "z"}, dry_run=True)
        == 20
    )
    assert hash_index.fetch("1")["status"] == "draft"  # unchanged
    with pytest.raises(ValueError):
        hash_index.update_by_filter(Tag("cat") == "a", {})


# --------------------------------------------------------------------------- #
# batched drop_documents / drop_keys
# --------------------------------------------------------------------------- #
def test_drop_documents_batched(hash_index):
    hash_index.load(_hash_data(30), id_field="id")
    dropped = hash_index.drop_documents([str(i) for i in range(20)], batch_size=7)
    assert dropped == 20
    assert hash_index.query(CountQuery(Num("n") >= 0)) == 10


def test_drop_keys_batched(hash_index):
    keys = hash_index.load(_hash_data(30), id_field="id")
    dropped = hash_index.drop_keys(keys[:20], batch_size=7)
    assert dropped == 20
    assert hash_index.query(CountQuery(Num("n") >= 0)) == 10


# --------------------------------------------------------------------------- #
# async coverage
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_delete_and_update_by_filter(worker_id, async_client):
    index = AsyncSearchIndex(
        schema=_schema(f"bulk_ah_{worker_id}", "hash"), redis_client=async_client
    )
    await index.create(overwrite=True, drop=True)
    try:
        await index.load(_hash_data(120), id_field="id")

        # guard
        with pytest.raises(ValueError):
            await index.delete_by_filter(None)

        updated = await index.update_by_filter(
            Tag("cat") == "b", {"status": "live"}, batch_size=25
        )
        assert updated == 60
        doc = await index.fetch("0")
        assert doc["status"] == "live" and doc["keep"] == "orig"

        deleted = await index.delete_by_filter(Num("n") < 60, batch_size=25)
        assert deleted == 60
        assert await index.query(CountQuery(Num("n") < 60)) == 0
    finally:
        await index.delete(drop=True)
