"""Unit tests for SearchIndex/AsyncSearchIndex.drop_keys using UNLINK (issue #600)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.schema import IndexSchema


def _schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {"name": "drop_keys_test", "prefix": "drop_keys_test"},
            "fields": [{"name": "id", "type": "tag"}],
        }
    )


class TestDropKeysUsesUnlink:
    """SearchIndex.drop_keys should issue UNLINK, not DEL.

    UNLINK reclaims memory on a background thread; DEL reclaims on the main
    thread and stalls the server when dropping a large key set (for example,
    scope-targeted SemanticCache invalidation).
    """

    def test_single_key_calls_unlink(self):
        client = MagicMock()
        client.unlink.return_value = 1
        client.delete.return_value = 1
        index = SearchIndex(schema=_schema(), redis_client=client)

        result = index.drop_keys("drop_keys_test:1")

        assert result == 1
        client.unlink.assert_called_once_with("drop_keys_test:1")
        client.delete.assert_not_called()

    def test_list_of_keys_calls_unlink(self):
        client = MagicMock()
        client.unlink.return_value = 3
        client.delete.return_value = 3
        index = SearchIndex(schema=_schema(), redis_client=client)

        keys = ["drop_keys_test:1", "drop_keys_test:2", "drop_keys_test:3"]
        result = index.drop_keys(keys)

        assert result == 3
        client.unlink.assert_called_once_with(*keys)
        client.delete.assert_not_called()


class TestAsyncDropKeysUsesUnlink:
    """AsyncSearchIndex.drop_keys should issue UNLINK, not DEL."""

    @pytest.mark.asyncio
    async def test_single_key_calls_unlink(self):
        client = MagicMock()
        client.unlink = AsyncMock(return_value=1)
        client.delete = AsyncMock(return_value=1)
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        result = await index.drop_keys("drop_keys_test:1")

        assert result == 1
        client.unlink.assert_awaited_once_with("drop_keys_test:1")
        client.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_list_of_keys_calls_unlink(self):
        client = MagicMock()
        client.unlink = AsyncMock(return_value=3)
        client.delete = AsyncMock(return_value=3)
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        keys = ["drop_keys_test:1", "drop_keys_test:2", "drop_keys_test:3"]
        result = await index.drop_keys(keys)

        assert result == 3
        client.unlink.assert_awaited_once_with(*keys)
        client.delete.assert_not_awaited()
