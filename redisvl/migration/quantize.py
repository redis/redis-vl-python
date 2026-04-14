"""Pipelined vector quantization helpers.

Provides pipeline-read, convert, and pipeline-write functions that replace
the per-key HGET loop with batched pipeline operations.
"""

from typing import Any, Dict, List, Optional

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
