from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from redisvl.index import AsyncSearchIndex
from redisvl.migration.utils import schemas_equal
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.types import AsyncRedisClient


async def async_list_indexes(
    *, redis_url: Optional[str] = None, redis_client: Optional[AsyncRedisClient] = None
) -> List[str]:
    """List all search indexes in Redis (async version)."""
    if redis_client is None:
        if not redis_url:
            raise ValueError("Must provide either redis_url or redis_client")
        redis_client = await RedisConnectionFactory._get_aredis_connection(
            redis_url=redis_url
        )
    index = AsyncSearchIndex.from_dict(
        {"index": {"name": "__redisvl_migration_helper__"}, "fields": []},
        redis_client=redis_client,
    )
    return await index.listall()


async def async_wait_for_index_ready(
    index: AsyncSearchIndex,
    *,
    timeout_seconds: int = 1800,
    poll_interval_seconds: float = 0.5,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[Dict[str, Any], float]:
    """Wait for index to finish indexing all documents (async version).

    Args:
        index: The AsyncSearchIndex to monitor.
        timeout_seconds: Maximum time to wait.
        poll_interval_seconds: How often to check status.
        progress_callback: Optional callback(indexed_docs, total_docs, percent).
    """
    start = time.perf_counter()
    deadline = start + timeout_seconds
    latest_info = await index.info()

    stable_ready_checks = 0
    while time.perf_counter() < deadline:
        latest_info = await index.info()
        indexing = latest_info.get("indexing")
        percent_indexed = latest_info.get("percent_indexed")

        if percent_indexed is not None or indexing is not None:
            ready = float(percent_indexed or 0) >= 1.0 and not bool(indexing)
            if progress_callback:
                total_docs = int(latest_info.get("num_docs", 0))
                pct = float(percent_indexed or 0)
                indexed_docs = int(total_docs * pct)
                progress_callback(indexed_docs, total_docs, pct * 100)
        else:
            current_docs = latest_info.get("num_docs")
            if current_docs is None:
                ready = True
            else:
                if stable_ready_checks == 0:
                    stable_ready_checks = int(current_docs)
                    await asyncio.sleep(poll_interval_seconds)
                    continue
                ready = int(current_docs) == stable_ready_checks

        if ready:
            return latest_info, round(time.perf_counter() - start, 3)

        await asyncio.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Index {index.schema.index.name} did not become ready within {timeout_seconds} seconds"
    )


async def async_current_source_matches_snapshot(
    index_name: str,
    expected_schema: Dict[str, Any],
    *,
    redis_url: Optional[str] = None,
    redis_client: Optional[AsyncRedisClient] = None,
) -> bool:
    """Check if current source schema matches the snapshot (async version)."""
    current_index = await AsyncSearchIndex.from_existing(
        index_name,
        redis_url=redis_url,
        redis_client=redis_client,
    )
    return schemas_equal(current_index.schema.to_dict(), expected_schema)
