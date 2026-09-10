"""Regression tests: ``paginate`` must not stop on a page whose matches were dropped.

``process_results`` deliberately drops matched documents whose field payload came
back missing (a key that expires or is updated mid-query is returned as a matched
id with a ``nil`` field array, and is still counted in the server's total), so a
page can be empty while the server still has matches to report. ``paginate``
previously terminated on ``if not results: break``, conflating "no more matches"
with "matches we could not materialize" and silently discarding every remaining
page.

The live race is not reproducible on demand, so these tests serve canned
``FT.SEARCH`` replies from a fake ``index.search``, keyed by the paging offset the
query carried. That seam runs the real ``_query`` -> ``process_results`` ->
``SearchResults`` chain, so the ``dropped_count`` the fix depends on is produced by
production code rather than hand-constructed. Every fake bounds its request count,
so a termination regression fails loudly instead of hanging the suite.
"""

import pytest
from redis.commands.search.result import Result

from redisvl.index import AsyncSearchIndex, SearchIndex, SearchResults
from redisvl.query import CountQuery, VectorQuery
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


def _healthy(doc_id):
    """A raw FT.SEARCH reply fragment for a match that carries its fields."""
    return [doc_id, ["vector_distance", "0.1", "brand", "Nike"]]


def _race_victim(doc_id):
    """A matched id whose field array came back nil -- the race victim.

    redis-py collapses the nil to an empty field set, leaving a ``Document`` whose
    ``__dict__`` is only ``{"id": ..., "payload": None}``. For a vector query
    (which always projects ``vector_distance``) ``process_results`` detects the
    missing payload and drops the doc, incrementing ``dropped_count``.
    """
    return [doc_id, None]


def _page(*fragments):
    flat = [item for fragment in fragments for item in fragment]
    return Result([len(fragments), *flat], True)


EMPTY_PAGE = Result([0], True)

# One pass over every drop shape that matters:
#
#   offset 0  both matches dropped   -> a LEADING dropped page must not terminate
#   offset 2  both healthy           -> carries offset 0's 2 drops  (dropped_count 2)
#   offset 4  both matches dropped   -> a second carry episode, forcing the reset
#   offset 6  1 healthy + 1 dropped  -> own 1 drop plus carried 2   (dropped_count 3)
#   offset 8  empty                  -> the genuine end of the result set
#
# The dropped_count sequence [2, 3] is what pins the accounting: losing the fold
# gives [0, 1], losing the reset gives [2, 5], and assigning instead of adding
# gives [2, 2].
DROP_SHAPES = {
    0: _page(_race_victim("doc:1"), _race_victim("doc:2")),
    2: _page(_healthy("doc:3"), _healthy("doc:4")),
    4: _page(_race_victim("doc:5"), _race_victim("doc:6")),
    6: _page(_healthy("doc:7"), _race_victim("doc:8")),
    8: EMPTY_PAGE,
}


def _fake_search(pages, seen):
    """Serve canned FT.SEARCH replies keyed by the query's paging offset.

    Keying by offset (rather than popping a queue) means a failure to advance
    shows up as repeated documents or a wedge, not as a silently different page.
    ``seen`` records the request order and bounds runaway loops. It reads
    redis-py's private ``Query._offset``, which ``paging()`` sets.
    """

    def _search(query, query_params=None):
        seen.append(query._offset)
        assert len(seen) <= 20, f"paginate made {len(seen)} requests; likely wedged"
        return pages.get(query._offset, EMPTY_PAGE)

    return _search


def test_paginate_continues_past_dropped_pages_and_accounts_for_their_drops():
    index = SearchIndex(_schema())
    seen: list[int] = []
    index.search = _fake_search(DROP_SHAPES, seen)  # type: ignore[method-assign]

    batches = list(index.paginate(_query(), page_size=2))

    # Iteration reached the end rather than stopping at offset 0's dropped page.
    assert [[doc["id"] for doc in batch] for batch in batches] == [
        ["doc:3", "doc:4"],
        ["doc:7"],
    ]
    # Never an empty batch, and the completeness metadata survives the generator.
    assert all(batch for batch in batches)
    assert all(isinstance(batch, SearchResults) for batch in batches)
    # Drops from skipped pages ride along on the next batch the caller sees.
    assert [batch.dropped_count for batch in batches] == [2, 3]
    assert [batch.complete for batch in batches] == [False, False]
    assert seen == [0, 2, 4, 6, 8]


@pytest.mark.asyncio
async def test_async_paginate_continues_past_dropped_pages_and_accounts_for_their_drops():
    """Async mirror: ``AsyncSearchIndex.paginate`` is a hand-duplicated copy."""
    index = AsyncSearchIndex(_schema())
    seen: list[int] = []
    sync_search = _fake_search(DROP_SHAPES, seen)

    async def _search(query, query_params=None):
        return sync_search(query, query_params)

    index.search = _search  # type: ignore[method-assign]

    batches = [batch async for batch in index.paginate(_query(), page_size=2)]

    assert [[doc["id"] for doc in batch] for batch in batches] == [
        ["doc:3", "doc:4"],
        ["doc:7"],
    ]
    assert all(batch for batch in batches)
    assert all(isinstance(batch, SearchResults) for batch in batches)
    assert [batch.dropped_count for batch in batches] == [2, 3]
    assert [batch.complete for batch in batches] == [False, False]
    assert seen == [0, 2, 4, 6, 8]


def test_paginate_rejects_count_query():
    """CountQuery returns a match count, not documents, so it cannot paginate.

    ``process_results`` returns a bare ``int`` for it, which used to make
    ``paginate`` yield that integer forever.
    """
    index = SearchIndex(_schema())
    with pytest.raises(TypeError, match="CountQuery cannot be paginated"):
        list(index.paginate(CountQuery("*"), page_size=2))


@pytest.mark.asyncio
async def test_async_paginate_rejects_count_query():
    index = AsyncSearchIndex(_schema())
    with pytest.raises(TypeError, match="CountQuery cannot be paginated"):
        [batch async for batch in index.paginate(CountQuery("*"), page_size=2)]


def test_paginate_validates_page_size():
    index = SearchIndex(_schema())
    with pytest.raises(TypeError, match="page_size must be an integer"):
        list(index.paginate(_query(), page_size="2"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="page_size must be greater than 0"):
        list(index.paginate(_query(), page_size=0))
