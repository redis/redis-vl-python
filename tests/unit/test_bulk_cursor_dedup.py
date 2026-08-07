"""Unit tests for cursor key de-duplication in bulk update (issue #577).

An ``FT.AGGREGATE ... WITHCURSOR`` cursor walks ascending internal document ids
rather than taking a snapshot, so it can hand back a key it already returned.
``_iter_keys_by_filter`` de-duplicates across pages so ``update_by_filter``
writes each document once. Reproducing real re-emission needs a live server and
luck, so these replay canned pages instead: the property under test is the
iterator's loop, not the server. The live multi-batch tests in
``tests/integration/test_bulk_operations.py`` cover the real wire shape.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from redis.commands.search.aggregation import Cursor

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema

# One cursor walk exercising every repeat shape, so a single test per interface
# covers them all:
#   page 2 repeats page 1 wholesale -> yields nothing, must not end the walk
#   page 3 repeats within itself and across pages
#   page 4 is a trailing all-repeat -> must not surface as an empty batch
CURSOR_PAGES = [["k:1", "k:2"], ["k:1", "k:2"], ["k:3", "k:3", "k:1"], ["k:2", "k:3"]]
EXPECTED_BATCHES = [["k:1", "k:2"], ["k:3"]]


def _schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {"name": "bulk_dedupe", "prefix": "bulk_dedupe"},
            "fields": [{"name": "cat", "type": "tag"}],
        }
    )


class _ReplayCursor:
    """Replays canned FT.AGGREGATE pages in place of ``client.ft(name)``.

    Assert ``reads``: it proves the canned pages were actually served, so a seam
    that stopped intercepting cannot leave the test passing vacuously.
    """

    def __init__(self, pages):
        self._pages = list(pages)
        self.reads = 0

    def _next_page(self):
        assert self._pages, "unexpected extra FT.AGGREGATE call"
        rows = self._pages.pop(0)
        self.reads += 1
        # Rows arrive from the wire as bytes (this exercises the convert_bytes
        # path in _agg_row_to_key); cursor id 0 means the server released it.
        return SimpleNamespace(
            rows=[[b"__key", key.encode()] for key in rows],
            cursor=Cursor(1 if self._pages else 0),
        )

    def aggregate(self, request):
        return self._next_page()


class _AsyncReplayCursor(_ReplayCursor):
    async def aggregate(self, request):
        return self._next_page()


def _index(index_cls, replay_cls, drains):
    """Build an index whose client serves `drains` independent canned walks."""
    replays = [replay_cls(CURSOR_PAGES) for _ in range(drains)]
    pending = iter(replays)
    client = MagicMock()
    client.ft.side_effect = lambda name: next(pending)
    return index_cls(schema=_schema(), redis_client=client), replays


def test_iter_keys_by_filter_dedupes_repeated_cursor_rows():
    index, (first, second) = _index(SearchIndex, _ReplayCursor, drains=2)

    assert list(index._iter_keys_by_filter(Tag("cat") == "a", 2)) == EXPECTED_BATCHES
    assert first.reads == 4  # every page served; repeats did not end the walk

    # De-dup state must not leak between calls, or a second update over the same
    # match set would silently write nothing.
    assert list(index._iter_keys_by_filter(Tag("cat") == "a", 2)) == EXPECTED_BATCHES
    assert second.reads == 4


@pytest.mark.asyncio
async def test_async_iter_keys_by_filter_dedupes_repeated_cursor_rows():
    index, (replay,) = _index(AsyncSearchIndex, _AsyncReplayCursor, drains=1)

    batches = [
        batch async for batch in index._iter_keys_by_filter(Tag("cat") == "a", 2)
    ]

    assert batches == EXPECTED_BATCHES
    assert replay.reads == 4
