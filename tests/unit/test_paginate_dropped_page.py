"""Regression tests: ``paginate`` must not stop on a page whose matches were dropped.

``process_results`` deliberately drops matched documents whose field payload came
back missing (the Redis 8.8+ background-WORKERS TTL/expiry race), so a page can
be empty while the server still has matches to report. ``paginate`` previously
terminated on ``if not results: break``, conflating "no more matches" with
"matches we could not materialize" and silently discarding every remaining page.

The live race is not reproducible on demand, so these tests serve canned
``FT.SEARCH`` replies from a fake ``index.search``, keyed by the paging offset the
query carried. That seam runs the real ``_query`` -> ``process_results`` ->
``SearchResults`` chain, so the ``dropped_count`` the fix depends on is produced
by production code rather than hand-constructed.
"""

import pytest
from redis.commands.search.result import Result

from redisvl.index import AsyncSearchIndex, SearchIndex, SearchResults
from redisvl.index.index import _fold_carried_drops, _page_had_matches
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
    """A matched id whose field array came back nil -- the expiry race victim.

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


def _install_pages(index, pages, seen):
    """Serve canned FT.SEARCH replies keyed by the query's paging offset.

    Keying by offset (rather than popping a queue) means a failure to advance the
    offset shows up as repeated documents or a wedge, not as a silently different
    page. ``seen`` records the request order and bounds runaway loops. It reads
    redis-py's private ``Query._offset``, which ``paging()`` sets.
    """

    def _fake_search(query, query_params=None):
        seen.append(query._offset)
        assert len(seen) <= 20, f"paginate made {len(seen)} requests; likely wedged"
        return pages.get(query._offset, EMPTY_PAGE)

    index.search = _fake_search  # type: ignore[method-assign]


# Three pages of matches where the whole middle page was dropped: the server
# matched doc:3 and doc:4 at offset 2 and neither could be materialized. Offset 6
# is the genuine end of the result set.
DROPPED_MIDDLE_PAGES = {
    0: _page(_healthy("doc:1"), _healthy("doc:2")),
    2: _page(_race_victim("doc:3"), _race_victim("doc:4")),
    4: _page(_healthy("doc:5"), _healthy("doc:6")),
    6: EMPTY_PAGE,
}


def test_paginate_continues_past_fully_dropped_page():
    """A page whose matches were all dropped must not end iteration."""
    index = SearchIndex(_schema())
    seen: list[int] = []
    _install_pages(index, DROPPED_MIDDLE_PAGES, seen)

    batches = list(index.paginate(_query(), page_size=2))

    ids = [doc["id"] for batch in batches for doc in batch]
    assert ids == ["doc:1", "doc:2", "doc:5", "doc:6"]
    # The dropped page is skipped, not yielded: every batch stays non-empty.
    assert all(batch for batch in batches)
    # Offset advanced across the dropped page so iteration made progress.
    assert seen == [0, 2, 4, 6]


def test_paginate_carries_dropped_count_of_skipped_page_forward():
    """A skipped page's drops must surface on the next yielded batch.

    Otherwise the batches the caller actually sees all report ``complete is
    True`` while matched documents went missing from the stream.
    """
    index = SearchIndex(_schema())
    _install_pages(index, DROPPED_MIDDLE_PAGES, [])

    batches = list(index.paginate(_query(), page_size=2))

    assert batches[0].dropped_count == 0
    assert batches[0].complete is True
    # doc:3 and doc:4 were dropped with their page; the count rides along here.
    assert batches[1].dropped_count == 2
    assert batches[1].complete is False


def test_paginate_yields_partially_dropped_page_with_its_count():
    """A page with survivors AND drops is yielded, carrying its own count."""
    index = SearchIndex(_schema())
    _install_pages(
        index,
        {
            0: _page(_healthy("doc:1"), _race_victim("doc:2")),
            2: EMPTY_PAGE,
        },
        [],
    )

    batches = list(index.paginate(_query(), page_size=2))

    assert len(batches) == 1
    assert [doc["id"] for doc in batches[0]] == ["doc:1"]
    # Batches must remain SearchResults, or the completeness contract is severed.
    assert isinstance(batches[0], SearchResults)
    assert batches[0].dropped_count == 1
    assert batches[0].complete is False


def test_paginate_stops_when_server_reports_no_matches():
    """An empty page with nothing dropped is still the end of the result set."""
    index = SearchIndex(_schema())
    seen: list[int] = []
    _install_pages(index, {0: _page(_healthy("doc:1")), 1: EMPTY_PAGE}, seen)

    batches = list(index.paginate(_query(), page_size=1))

    assert [doc["id"] for batch in batches for doc in batch] == ["doc:1"]
    # Stopped on the empty page -- no third request.
    assert seen == [0, 1]


def test_paginate_zero_matches_yields_nothing():
    """An empty first page terminates immediately rather than looping."""
    index = SearchIndex(_schema())
    seen: list[int] = []
    _install_pages(index, {}, seen)

    assert list(index.paginate(_query(), page_size=10)) == []
    assert seen == [0]


def test_paginate_terminates_on_trailing_dropped_page():
    """A fully-dropped final page costs one extra request, then terminates."""
    index = SearchIndex(_schema())
    seen: list[int] = []
    _install_pages(
        index,
        {
            0: _page(_healthy("doc:1"), _healthy("doc:2")),
            2: _page(_race_victim("doc:3")),
            4: EMPTY_PAGE,
        },
        seen,
    )

    batches = list(index.paginate(_query(), page_size=2))

    assert [doc["id"] for batch in batches for doc in batch] == ["doc:1", "doc:2"]
    assert seen == [0, 2, 4]
    # Trailing drops have no later batch to ride on; process_results logs them.
    assert batches[0].dropped_count == 0


def test_paginate_terminates_when_every_page_is_dropped():
    """All-dropped pages must terminate once the server runs out of matches."""
    index = SearchIndex(_schema())
    seen: list[int] = []
    _install_pages(
        index,
        {
            0: _page(_race_victim("doc:1"), _race_victim("doc:2")),
            2: _page(_race_victim("doc:3"), _race_victim("doc:4")),
            4: EMPTY_PAGE,
        },
        seen,
    )

    assert list(index.paginate(_query(), page_size=2)) == []
    assert seen == [0, 2, 4]


def test_paginate_rejects_count_query():
    """CountQuery returns a match count, not documents, so it cannot paginate.

    ``process_results`` returns a bare ``int`` for it, which used to make
    ``paginate`` yield that integer forever.
    """
    index = SearchIndex(_schema())
    with pytest.raises(TypeError, match="CountQuery cannot be paginated"):
        list(index.paginate(CountQuery("*"), page_size=2))


def test_paginate_validates_page_size():
    index = SearchIndex(_schema())
    with pytest.raises(TypeError, match="page_size must be an integer"):
        list(index.paginate(_query(), page_size="2"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="page_size must be greater than 0"):
        list(index.paginate(_query(), page_size=0))


def test_paginate_tolerates_plain_list_pages():
    """``paginate`` still works if ``_query`` is overridden to return a plain list.

    No production query path does this -- every one returns ``SearchResults`` --
    but the termination check must not assume the metadata is present.
    """
    index = SearchIndex(_schema())
    responses: list[list[dict]] = [[{"id": "doc:1"}], []]
    index._query = lambda query: responses.pop(0) if responses else []  # type: ignore[method-assign]

    batches = list(index.paginate(_query(), page_size=1))

    assert [doc["id"] for batch in batches for doc in batch] == ["doc:1"]


# ---------------------------------------------------------------------------
# Async mirrors. AsyncSearchIndex.paginate is a hand-duplicated copy of the sync
# logic, so it is the natural place for the two to drift.
# ---------------------------------------------------------------------------


def _install_async_pages(index, pages, seen):
    async def _fake_search(query, query_params=None):
        seen.append(query._offset)
        assert len(seen) <= 20, f"paginate made {len(seen)} requests; likely wedged"
        return pages.get(query._offset, EMPTY_PAGE)

    index.search = _fake_search  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_async_paginate_continues_past_fully_dropped_page():
    """Async mirror: the dropped middle page must not truncate iteration."""
    index = AsyncSearchIndex(_schema())
    seen: list[int] = []
    _install_async_pages(index, DROPPED_MIDDLE_PAGES, seen)

    batches = [batch async for batch in index.paginate(_query(), page_size=2)]

    ids = [doc["id"] for batch in batches for doc in batch]
    assert ids == ["doc:1", "doc:2", "doc:5", "doc:6"]
    assert all(batch for batch in batches)
    assert seen == [0, 2, 4, 6]
    # Skipped page's drops carried onto the following batch.
    assert batches[1].dropped_count == 2


@pytest.mark.asyncio
async def test_async_paginate_stops_when_server_reports_no_matches():
    index = AsyncSearchIndex(_schema())
    seen: list[int] = []
    _install_async_pages(index, {0: _page(_healthy("doc:1")), 1: EMPTY_PAGE}, seen)

    batches = [batch async for batch in index.paginate(_query(), page_size=1)]

    assert [doc["id"] for batch in batches for doc in batch] == ["doc:1"]
    assert seen == [0, 1]


@pytest.mark.asyncio
async def test_async_paginate_terminates_when_every_page_is_dropped():
    index = AsyncSearchIndex(_schema())
    seen: list[int] = []
    _install_async_pages(
        index,
        {
            0: _page(_race_victim("doc:1"), _race_victim("doc:2")),
            2: EMPTY_PAGE,
        },
        seen,
    )

    assert [batch async for batch in index.paginate(_query(), page_size=2)] == []
    assert seen == [0, 2]


@pytest.mark.asyncio
async def test_async_paginate_rejects_count_query():
    index = AsyncSearchIndex(_schema())
    with pytest.raises(TypeError, match="CountQuery cannot be paginated"):
        [batch async for batch in index.paginate(CountQuery("*"), page_size=2)]


@pytest.mark.asyncio
async def test_async_paginate_tolerates_plain_list_pages():
    index = AsyncSearchIndex(_schema())
    responses: list[list[dict]] = [[{"id": "doc:1"}], []]

    async def _fake_query(query):
        return responses.pop(0) if responses else []

    index._query = _fake_query  # type: ignore[method-assign]

    batches = [batch async for batch in index.paginate(_query(), page_size=1)]

    assert [doc["id"] for batch in batches for doc in batch] == ["doc:1"]


# ---------------------------------------------------------------------------
# Helper contracts, pinned directly so the intent survives a refactor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "results, expected",
    [
        (SearchResults([]), False),
        (SearchResults([{"id": "doc:1"}]), True),
        (SearchResults([], dropped_count=1), True),
        (SearchResults([{"id": "doc:1"}], dropped_count=1), True),
        ([], False),
        ([{"id": "doc:1"}], True),
        # process_results returns a bare int for CountQuery; must not raise.
        (0, False),
        (5, True),
        (None, False),
    ],
)
def test_page_had_matches(results, expected):
    assert _page_had_matches(results) is expected


def test_fold_carried_drops_adds_to_existing_count():
    results = SearchResults([{"id": "doc:1"}], dropped_count=1)
    _fold_carried_drops(results, 2)
    assert results.dropped_count == 3


def test_fold_carried_drops_noop_without_carry():
    results = SearchResults([{"id": "doc:1"}], dropped_count=1)
    _fold_carried_drops(results, 0)
    assert results.dropped_count == 1


def test_fold_carried_drops_ignores_plain_list():
    """A plain list carries no metadata; folding must not raise."""
    plain = [{"id": "doc:1"}]
    _fold_carried_drops(plain, 2)
    assert plain == [{"id": "doc:1"}]
