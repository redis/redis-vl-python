"""Regression tests: ``paginate`` must not stop on a page whose matches were dropped.

``process_results`` deliberately drops matched documents whose field payload came
back missing (the Redis 8.8+ background-WORKERS TTL/expiry race), so a page can
be empty while the server still has matches to report. ``paginate`` previously
terminated on ``if not results: break``, conflating "no more matches" with
"matches we could not materialize" and silently discarding every remaining page.

The live 8.8 race is not reproducible on demand, so these tests drive the
termination logic directly with canned pages, mirroring the canned-page style
used elsewhere for bulk cursor tests.
"""

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.index.index import SearchResults
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

sample_vector = [0.1, 0.1, 0.5, 0.15]


def _schema():
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "paginate_test",
                "prefix": "test",
                "storage_type": "hash",
            },
            "fields": [
                {
                    "name": "user_embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 4,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
                {"name": "brand", "type": "tag"},
            ],
        }
    )


def _query():
    return VectorQuery(
        vector=sample_vector,
        vector_field_name="user_embedding",
        return_fields=["brand"],
    )


def _doc(doc_id):
    return {"id": doc_id, "vector_distance": "0.1", "brand": "Nike"}


def _canned_pages():
    """Three pages of matches where the whole middle page was dropped.

    Page 2 is empty but reports ``dropped_count=2``: the server matched two docs
    and neither could be materialized. Page 4 is the genuine end of the result
    set — empty with nothing dropped.
    """
    return [
        SearchResults([_doc("doc:1"), _doc("doc:2")]),
        SearchResults([], dropped_count=2),
        SearchResults([_doc("doc:5"), _doc("doc:6")]),
        SearchResults([]),
    ]


def _install_canned_query(index, pages, offsets):
    """Replace the index's query execution with a canned page sequence.

    Records the paging offset the query carried on each call so tests can assert
    the offset keeps advancing across a dropped page.
    """
    responses = list(pages)

    def _fake_query(query):
        offsets.append(query._offset)
        return responses.pop(0) if responses else SearchResults([])

    index._query = _fake_query  # type: ignore[method-assign]


def test_paginate_continues_past_fully_dropped_page():
    """A page whose matches were all dropped must not end iteration."""
    index = SearchIndex(_schema())
    offsets: list[int] = []
    _install_canned_query(index, _canned_pages(), offsets)

    batches = list(index.paginate(_query(), page_size=2))

    ids = [doc["id"] for batch in batches for doc in batch]
    assert ids == ["doc:1", "doc:2", "doc:5", "doc:6"]
    # The dropped page is skipped, not yielded: every batch stays non-empty.
    assert all(batch for batch in batches)
    # Offset advances across the dropped page so iteration makes progress.
    assert offsets == [0, 2, 4, 6]


def test_paginate_stops_when_server_reports_no_matches():
    """An empty page with nothing dropped is still the end of the result set."""
    index = SearchIndex(_schema())
    offsets: list[int] = []
    _install_canned_query(
        index,
        [SearchResults([_doc("doc:1")]), SearchResults([])],
        offsets,
    )

    batches = list(index.paginate(_query(), page_size=1))

    assert [doc["id"] for batch in batches for doc in batch] == ["doc:1"]
    # Stopped on the empty page -- no third request.
    assert offsets == [0, 1]


def test_paginate_terminates_when_every_page_is_dropped():
    """All-dropped pages must terminate once the server runs out of matches."""
    index = SearchIndex(_schema())
    offsets: list[int] = []
    _install_canned_query(
        index,
        [
            SearchResults([], dropped_count=2),
            SearchResults([], dropped_count=2),
            SearchResults([]),
        ],
        offsets,
    )

    assert list(index.paginate(_query(), page_size=2)) == []
    assert offsets == [0, 2, 4]


def test_paginate_tolerates_plain_list_pages():
    """A query path returning a plain ``list`` (no metadata) behaves as before."""
    index = SearchIndex(_schema())
    offsets: list[int] = []
    _install_canned_query(index, [[_doc("doc:1")], []], offsets)

    batches = list(index.paginate(_query(), page_size=1))

    assert [doc["id"] for batch in batches for doc in batch] == ["doc:1"]


@pytest.mark.asyncio
async def test_async_paginate_continues_past_fully_dropped_page():
    """Async mirror: the dropped middle page must not truncate iteration."""
    index = AsyncSearchIndex(_schema())
    offsets: list[int] = []
    responses = _canned_pages()

    async def _fake_query(query):
        offsets.append(query._offset)
        return responses.pop(0) if responses else SearchResults([])

    index._query = _fake_query  # type: ignore[method-assign]

    batches = [batch async for batch in index.paginate(_query(), page_size=2)]

    ids = [doc["id"] for batch in batches for doc in batch]
    assert ids == ["doc:1", "doc:2", "doc:5", "doc:6"]
    assert all(batch for batch in batches)
    assert offsets == [0, 2, 4, 6]


@pytest.mark.asyncio
async def test_async_paginate_stops_when_server_reports_no_matches():
    index = AsyncSearchIndex(_schema())
    offsets: list[int] = []
    responses = [SearchResults([_doc("doc:1")]), SearchResults([])]

    async def _fake_query(query):
        offsets.append(query._offset)
        return responses.pop(0) if responses else SearchResults([])

    index._query = _fake_query  # type: ignore[method-assign]

    batches = [batch async for batch in index.paginate(_query(), page_size=1)]

    assert [doc["id"] for batch in batches for doc in batch] == ["doc:1"]
    assert offsets == [0, 1]
