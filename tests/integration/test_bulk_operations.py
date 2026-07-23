"""Integration tests for bulk delete/update by filter (RAAE-1326)."""

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.index.index import BulkResult
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
            "index": {"name": name, "prefix": name, "storage_type": storage_type},
            "fields": BULK_FIELDS,
        }
    )


def _data(n=40):
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
# BulkResult contract
# --------------------------------------------------------------------------- #
def test_bulk_result_fields(hash_index):
    hash_index.load(_data(), id_field="id")
    result = hash_index.drop_by_filter(Tag("cat") == "a")
    assert isinstance(result, BulkResult)
    assert result.matched == 20
    assert result.processed == 20
    assert result.completed is True
    assert result.dry_run is False


# --------------------------------------------------------------------------- #
# drop_by_filter
# --------------------------------------------------------------------------- #
def test_drop_by_filter_removes_only_matches(hash_index):
    hash_index.load(_data(), id_field="id")
    result = hash_index.drop_by_filter(Tag("cat") == "a", batch_size=10)
    assert result.processed == 20
    assert hash_index.query(CountQuery(Tag("cat") == "a")) == 0
    assert hash_index.query(CountQuery(Tag("cat") == "b")) == 20


def test_drop_by_filter_json(json_index):
    json_index.load(_data(30), id_field="id")  # 15 'a' + 15 'b'
    result = json_index.drop_by_filter(Tag("cat") == "a", batch_size=10)
    assert result.processed == 15
    assert json_index.query(CountQuery(Tag("cat") == "a")) == 0
    assert json_index.query(CountQuery(Tag("cat") == "b")) == 15


def test_drop_by_filter_dry_run_does_not_mutate(hash_index):
    hash_index.load(_data(), id_field="id")
    result = hash_index.drop_by_filter(Num("n") < 10, dry_run=True)
    assert result.dry_run is True
    assert result.matched == 10
    assert result.processed == 10
    assert hash_index.query(CountQuery(Num("n") < 10)) == 10  # nothing removed


def test_drop_by_filter_reports_progress(hash_index):
    hash_index.load(_data(), id_field="id")
    progress = []
    hash_index.drop_by_filter(
        Tag("cat") == "a",
        batch_size=5,
        on_progress=lambda p, t: progress.append((p, t)),
    )
    assert progress
    processed = [p for p, _ in progress]
    assert processed == sorted(processed)
    assert processed[-1] == 20
    assert all(total == 20 for _, total in progress)  # matched total passed through


@pytest.mark.parametrize("bad_filter", [None, "*", ""])
def test_drop_by_filter_guards_match_all(hash_index, bad_filter):
    hash_index.load(_data(), id_field="id")
    with pytest.raises(ValueError):
        hash_index.drop_by_filter(bad_filter)
    assert hash_index.query(CountQuery(Num("n") >= 0)) == 40


def test_drop_by_filter_allow_all_override(hash_index):
    hash_index.load(_data(), id_field="id")
    result = hash_index.drop_by_filter("*", allow_all=True, batch_size=10)
    assert result.processed == 40
    assert hash_index.query(CountQuery(Num("n") >= 0)) == 0


# --------------------------------------------------------------------------- #
# update_by_filter
# --------------------------------------------------------------------------- #
def test_update_by_filter_hash_is_partial(hash_index):
    hash_index.load(_data(), id_field="id")
    result = hash_index.update_by_filter(
        Tag("cat") == "b", {"status": "published"}, batch_size=10
    )
    assert result.processed == 20
    doc = hash_index.fetch("0")  # id 0 -> cat "b"
    assert doc["status"] == "published"
    assert doc["keep"] == "orig"  # untouched field preserved
    assert doc["cat"] == "b"
    assert hash_index.fetch("1")["status"] == "draft"  # non-matching untouched


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


def test_update_by_filter_json_none_deletes_path(json_index):
    json_index.load(
        [{"id": "x", "cat": "a", "status": "draft", "n": 1, "drop_me": "gone"}],
        id_field="id",
    )
    json_index.update_by_filter(Tag("cat") == "a", {"drop_me": None})
    doc = json_index.fetch("x")
    assert "drop_me" not in doc  # None deletes the path (RFC 7396)


def test_update_by_filter_spans_multiple_batches_hash(hash_index):
    # more docs than batch_size to exercise cursor staging + multi-batch writes
    hash_index.load(_data(120), id_field="id")
    progress = []
    result = hash_index.update_by_filter(
        Tag("cat") == "a",
        {"status": "x"},
        batch_size=25,
        on_progress=lambda p, t: progress.append((p, t)),
    )
    assert result.processed == 60
    assert len([p for p, _ in progress]) >= 2  # spanned multiple batches
    assert progress[-1] == (60, 60)


def test_update_by_filter_spans_multiple_batches_json(json_index):
    json_index.load(_data(120), id_field="id")
    result = json_index.update_by_filter(
        Tag("cat") == "b", {"status": "live"}, batch_size=25
    )
    assert result.processed == 60
    # spot-check a couple of docs across batches
    assert json_index.fetch("0")["status"] == "live"
    assert json_index.fetch("118")["status"] == "live"


def test_update_by_filter_skips_concurrently_deleted_key(hash_index, client):
    # Simulate the resolve->write race: a matched key is deleted before the
    # write lands. The guard must skip it (not recreate a partial doc) and not
    # count it in processed.
    hash_index.load(_data(4), id_field="id")  # ids 1,3 -> 'a'; 0,2 -> 'b'
    a_keys = [hash_index.key("1"), hash_index.key("3")]
    client.unlink(a_keys[0])  # id "1" vanishes mid-flight

    written = hash_index._apply_update_batch(a_keys, {"status": "x"})

    assert written == 1  # deleted key skipped, not counted
    assert not client.exists(a_keys[0])  # NOT recreated as a partial doc
    assert hash_index.fetch("3")["status"] == "x"  # survivor updated


def test_update_by_filter_json_skips_missing_key(json_index, client):
    json_index.load([{"id": "x", "cat": "a", "status": "draft", "n": 1}], id_field="id")
    missing = json_index.key("does-not-exist")
    written = json_index._apply_update_batch(
        [json_index.key("x"), missing], {"status": "done"}
    )
    assert written == 1
    assert not client.exists(missing)  # not recreated
    assert json_index.fetch("x")["status"] == "done"


def test_update_by_filter_dry_run_and_empty_values(hash_index):
    hash_index.load(_data(), id_field="id")
    result = hash_index.update_by_filter(
        Tag("cat") == "a", {"status": "z"}, dry_run=True
    )
    assert result.dry_run is True and result.processed == 20
    assert hash_index.fetch("1")["status"] == "draft"  # unchanged
    with pytest.raises(ValueError):
        hash_index.update_by_filter(Tag("cat") == "a", {})


# --------------------------------------------------------------------------- #
# batched drop_documents / drop_keys
# --------------------------------------------------------------------------- #
def test_drop_documents_batched(hash_index):
    hash_index.load(_data(30), id_field="id")
    dropped = hash_index.drop_documents([str(i) for i in range(20)], batch_size=7)
    assert dropped == 20
    assert hash_index.query(CountQuery(Num("n") >= 0)) == 10


def test_drop_keys_batched(hash_index):
    keys = hash_index.load(_data(30), id_field="id")
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
        await index.load(_data(120), id_field="id")

        with pytest.raises(ValueError):
            await index.drop_by_filter(None)

        result = await index.update_by_filter(
            Tag("cat") == "b", {"status": "live"}, batch_size=25
        )
        assert result.processed == 60
        doc = await index.fetch("0")
        assert doc["status"] == "live" and doc["keep"] == "orig"

        result = await index.drop_by_filter(Num("n") < 60, batch_size=25)
        assert result.processed == 60
        assert await index.query(CountQuery(Num("n") < 60)) == 0
    finally:
        await index.delete(drop=True)
