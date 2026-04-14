"""Pipelined vector quantization helpers.

Provides pipeline-read, convert, and pipeline-write functions that replace
the per-key HGET loop with batched pipeline operations.

Also provides multi-worker orchestration for parallel quantization
using ThreadPoolExecutor (sync) or asyncio.gather (async).
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from redisvl.redis.utils import array_to_buffer, buffer_to_array


def pipeline_read_vectors(
    client: Any,
    keys: List[str],
    datatype_changes: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, bytes]]:
    """Pipeline-read vector fields from Redis for a batch of keys.

    Instead of N individual HGET calls (N round trips), uses a single
    pipeline with N*F HGET calls (1 round trip).

    Args:
        client: Redis client
        keys: List of Redis keys to read
        datatype_changes: {field_name: {"source", "target", "dims"}}

    Returns:
        {key: {field_name: original_bytes}} — only includes keys/fields
        that returned non-None data.
    """
    if not keys:
        return {}

    pipe = client.pipeline(transaction=False)
    # Track the order of pipelined calls: (key, field_name)
    call_order: List[tuple] = []
    field_names = list(datatype_changes.keys())

    for key in keys:
        for field_name in field_names:
            pipe.hget(key, field_name)
            call_order.append((key, field_name))

    results = pipe.execute()

    # Reassemble into {key: {field: bytes}}
    output: Dict[str, Dict[str, bytes]] = {}
    for (key, field_name), value in zip(call_order, results):
        if value is not None:
            if key not in output:
                output[key] = {}
            output[key][field_name] = value

    return output


def pipeline_write_vectors(
    client: Any,
    converted: Dict[str, Dict[str, bytes]],
) -> None:
    """Pipeline-write converted vectors to Redis.

    Args:
        client: Redis client
        converted: {key: {field_name: new_bytes}}
    """
    if not converted:
        return

    pipe = client.pipeline(transaction=False)
    for key, fields in converted.items():
        for field_name, data in fields.items():
            pipe.hset(key, field_name, data)
    pipe.execute()


def convert_vectors(
    originals: Dict[str, Dict[str, bytes]],
    datatype_changes: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, bytes]]:
    """Convert vector bytes from source dtype to target dtype.

    Args:
        originals: {key: {field_name: original_bytes}}
        datatype_changes: {field_name: {"source", "target", "dims"}}

    Returns:
        {key: {field_name: converted_bytes}}
    """
    converted: Dict[str, Dict[str, bytes]] = {}
    for key, fields in originals.items():
        converted[key] = {}
        for field_name, data in fields.items():
            change = datatype_changes.get(field_name)
            if not change:
                continue
            array = buffer_to_array(data, change["source"])
            new_bytes = array_to_buffer(array, change["target"])
            converted[key][field_name] = new_bytes
    return converted


logger = logging.getLogger(__name__)


@dataclass
class MultiWorkerResult:
    """Result from multi-worker quantization."""

    total_docs_quantized: int
    num_workers: int
    worker_results: List[Dict[str, Any]] = field(default_factory=list)


def split_keys(keys: List[str], num_workers: int) -> List[List[str]]:
    """Split keys into N contiguous slices for parallel processing.

    Args:
        keys: Full list of Redis keys
        num_workers: Number of workers

    Returns:
        List of key slices (some may be empty if keys < workers)
    """
    if not keys:
        return []
    n = len(keys)
    chunk_size = math.ceil(n / num_workers)
    return [keys[i : i + chunk_size] for i in range(0, n, chunk_size)]


def _worker_quantize(
    worker_id: int,
    redis_url: str,
    keys: List[str],
    datatype_changes: Dict[str, Dict[str, Any]],
    backup_path: str,
    index_name: str,
    batch_size: int,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """Single worker: dump originals + convert + write back.

    Each worker gets its own Redis connection and backup file shard.
    """
    from redisvl.migration.backup import VectorBackup
    from redisvl.redis.connection import RedisConnectionFactory

    client = RedisConnectionFactory.get_redis_connection(redis_url=redis_url)
    try:
        # Phase 1: Dump originals to backup shard
        backup = VectorBackup.create(
            path=backup_path,
            index_name=index_name,
            fields=datatype_changes,
            batch_size=batch_size,
        )

        total = len(keys)
        for batch_start in range(0, total, batch_size):
            batch_keys = keys[batch_start : batch_start + batch_size]
            originals = pipeline_read_vectors(client, batch_keys, datatype_changes)
            backup.write_batch(batch_start // batch_size, batch_keys, originals)
            if progress_callback:
                progress_callback(
                    "dump", worker_id, min(batch_start + batch_size, total)
                )

        backup.mark_dump_complete()

        # Phase 2: Convert + write from backup
        backup.start_quantize()
        docs_quantized = 0

        for batch_idx, (batch_keys, originals) in enumerate(backup.iter_batches()):
            converted = convert_vectors(originals, datatype_changes)
            if converted:
                pipeline_write_vectors(client, converted)
            backup.mark_batch_quantized(batch_idx)
            docs_quantized += len(batch_keys)
            if progress_callback:
                progress_callback("quantize", worker_id, docs_quantized)

        backup.mark_complete()
        return {"worker_id": worker_id, "docs": docs_quantized}
    finally:
        try:
            client.close()
        except Exception:
            pass


def multi_worker_quantize(
    redis_url: str,
    keys: List[str],
    datatype_changes: Dict[str, Dict[str, Any]],
    backup_dir: str,
    index_name: str,
    num_workers: int = 1,
    batch_size: int = 500,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> MultiWorkerResult:
    """Orchestrate multi-worker quantization.

    Splits keys across N workers, each with its own Redis connection
    and backup file shard. Uses ThreadPoolExecutor for parallelism.

    Args:
        redis_url: Redis connection URL
        keys: Full list of document keys to quantize
        datatype_changes: {field_name: {"source", "target", "dims"}}
        backup_dir: Directory for backup file shards
        index_name: Source index name
        num_workers: Number of parallel workers (default 1)
        batch_size: Keys per pipeline batch
        progress_callback: Optional callback(phase, worker_id, docs_done)

    Returns:
        MultiWorkerResult with total docs quantized and per-worker results
    """
    from pathlib import Path

    slices = split_keys(keys, num_workers)
    actual_workers = len(slices)

    if actual_workers == 0:
        return MultiWorkerResult(
            total_docs_quantized=0, num_workers=0, worker_results=[]
        )

    # Generate backup paths per worker
    safe_name = index_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    worker_backup_paths = [
        str(Path(backup_dir) / f"migration_backup_{safe_name}_worker{i}")
        for i in range(actual_workers)
    ]

    if actual_workers == 1:
        # Single worker — run directly, no ThreadPoolExecutor overhead
        result = _worker_quantize(
            worker_id=0,
            redis_url=redis_url,
            keys=slices[0],
            datatype_changes=datatype_changes,
            backup_path=worker_backup_paths[0],
            index_name=index_name,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
        return MultiWorkerResult(
            total_docs_quantized=result["docs"],
            num_workers=1,
            worker_results=[result],
        )

    # Multi-worker — ThreadPoolExecutor
    worker_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {}
        for i, key_slice in enumerate(slices):
            future = executor.submit(
                _worker_quantize,
                worker_id=i,
                redis_url=redis_url,
                keys=key_slice,
                datatype_changes=datatype_changes,
                backup_path=worker_backup_paths[i],
                index_name=index_name,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )
            futures[future] = i

        for future in as_completed(futures):
            result = future.result()  # raises if worker failed
            worker_results.append(result)

    total_docs = sum(r["docs"] for r in worker_results)
    return MultiWorkerResult(
        total_docs_quantized=total_docs,
        num_workers=actual_workers,
        worker_results=worker_results,
    )
