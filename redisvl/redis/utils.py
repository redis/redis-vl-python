import hashlib
import itertools
import time
from typing import Any, Dict, List, Optional, Union

from redis import RedisCluster
from redis import __version__ as redis_version
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.client import NEVER_DECODE, Pipeline
from redis.commands.search import AsyncSearch, Search
from redis.commands.search.commands import (
    CREATE_CMD,
    MAXTEXTFIELDS,
    NOFIELDS,
    NOFREQS,
    NOHL,
    NOOFFSETS,
    SEARCH_CMD,
    SKIPINITIALSCAN,
    STOPWORDS,
    TEMPORARY,
)
from redis.commands.search.field import Field

from redisvl.utils.redis_protocol import get_protocol_version

# Redis 5.x compatibility (6 fixed the import path)
if redis_version.startswith("5"):
    from redis.commands.search.indexDefinition import (  # type: ignore[import-untyped]
        IndexDefinition,
    )
else:
    from redis.commands.search.index_definition import (  # type: ignore[no-redef]
        IndexDefinition,
    )

from redis.commands.search.query import Query
from redis.commands.search.result import Result

from redisvl.utils.utils import lazy_import

# Lazy import numpy
np = lazy_import("numpy")

from redisvl.schema.fields import VectorDataType


def make_dict(values: List[Any]) -> Dict[Any, Any]:
    """Convert a list of objects into a dictionary"""
    i = 0
    di = {}
    while i < len(values) - 1:
        di[values[i]] = values[i + 1]
        i += 2
    return di


def convert_bytes(data: Any) -> Any:
    """Convert bytes data back to string"""
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except:
            return data
    if isinstance(data, dict):
        return {convert_bytes(key): convert_bytes(value) for key, value in data.items()}
    if isinstance(data, list):
        return [convert_bytes(item) for item in data]
    if isinstance(data, tuple):
        return tuple(convert_bytes(item) for item in data)
    return data


def array_to_buffer(array: List[float], dtype: str) -> bytes:
    """Convert a list of floats into a numpy byte string."""
    try:
        VectorDataType(dtype.upper())
    except ValueError:
        raise ValueError(
            f"Invalid data type: {dtype}. Supported types are: {[t.lower() for t in VectorDataType]}"
        )

    # Special handling for bfloat16 which requires explicit import from ml_dtypes
    if dtype.lower() == "bfloat16":
        from ml_dtypes import bfloat16

        return np.array(array, dtype=bfloat16).tobytes()

    return np.array(array, dtype=dtype.lower()).tobytes()


def buffer_to_array(buffer: bytes, dtype: str) -> List[Any]:
    """Convert bytes into into a list of numerics."""
    try:
        VectorDataType(dtype.upper())
    except ValueError:
        raise ValueError(
            f"Invalid data type: {dtype}. Supported types are: {[t.lower() for t in VectorDataType]}"
        )

    # Special handling for bfloat16 which requires explicit import from ml_dtypes
    # because otherwise the (lazily imported) numpy is unaware of the type
    if dtype.lower() == "bfloat16":
        from ml_dtypes import bfloat16

        return np.frombuffer(buffer, dtype=bfloat16).tolist()  # type: ignore[return-value]

    return np.frombuffer(buffer, dtype=dtype.lower()).tolist()  # type: ignore[return-value]


def hashify(content: str, extras: Optional[Dict[str, Any]] = None) -> str:
    """Create a secure hash of some arbitrary input text and optional dictionary."""
    if extras:
        extra_string = " ".join([str(k) + str(v) for k, v in sorted(extras.items())])
        content = content + extra_string
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def cluster_create_index(
    index_name: str,
    client: RedisCluster,
    fields: List[Field],
    no_term_offsets: bool = False,
    no_field_flags: bool = False,
    stopwords: Optional[List[str]] = None,
    definition: Optional[IndexDefinition] = None,
    max_text_fields=False,
    temporary=None,
    no_highlight: bool = False,
    no_term_frequencies: bool = False,
    skip_initial_scan: bool = False,
):
    """
    Creates the search index. The index must not already exist.

    For more information, see https://redis.io/commands/ft.create/

    Args:
        index_name: The name of the index to create.
        client: The redis client to use.
        fields: A list of Field objects.
        no_term_offsets: If `true`, term offsets will not be saved in the index.
        no_field_flags: If true, field flags that allow searching in specific fields
                        will not be saved.
        stopwords: If provided, the index will be created with this custom stopword
                   list. The list can be empty.
        definition: If provided, the index will be created with this custom index
                    definition.
        max_text_fields: If true, indexes will be encoded as if there were more than
                         32 text fields, allowing for additional fields beyond 32.
        temporary: Creates a lightweight temporary index which will expire after the
                   specified period of inactivity. The internal idle timer is reset
                   whenever the index is searched or added to.
        no_highlight: If true, disables highlighting support. Also implied by
                      `no_term_offsets`.
        no_term_frequencies: If true, term frequencies will not be saved in the
                             index.
        skip_initial_scan: If true, the initial scan and indexing will be skipped.

    """
    args = [CREATE_CMD, index_name]
    if definition is not None:
        args += definition.args
    if max_text_fields:
        args.append(MAXTEXTFIELDS)
    if temporary is not None and isinstance(temporary, int):
        args.append(TEMPORARY)
        args.append(str(temporary))
    if no_term_offsets:
        args.append(NOOFFSETS)
    if no_highlight:
        args.append(NOHL)
    if no_field_flags:
        args.append(NOFIELDS)
    if no_term_frequencies:
        args.append(NOFREQS)
    if skip_initial_scan:
        args.append(SKIPINITIALSCAN)
    if stopwords is not None and isinstance(stopwords, (list, tuple, set)):
        args += [STOPWORDS, str(len(stopwords))]
        if len(stopwords) > 0:
            args += list(stopwords)

    args.append("SCHEMA")
    try:
        args += list(itertools.chain(*(f.redis_args() for f in fields)))
    except TypeError:
        args += fields.redis_args()  # type: ignore

    default_node = client.get_default_node()
    return client.execute_command(*args, target_nodes=[default_node])


async def async_cluster_create_index(
    index_name: str,
    client: AsyncRedisCluster,
    fields: List[Field],
    no_term_offsets: bool = False,
    no_field_flags: bool = False,
    stopwords: Optional[List[str]] = None,
    definition: Optional[IndexDefinition] = None,
    max_text_fields=False,
    temporary=None,
    no_highlight: bool = False,
    no_term_frequencies: bool = False,
    skip_initial_scan: bool = False,
):
    """
    Creates the search index. The index must not already exist.

    For more information, see https://redis.io/commands/ft.create/

    Args:
        index_name: The name of the index to create.
        client: The redis client to use.
        fields: A list of Field objects.
        no_term_offsets: If `true`, term offsets will not be saved in the index.
        no_field_flags: If true, field flags that allow searching in specific fields
                        will not be saved.
        stopwords: If provided, the index will be created with this custom stopword
                   list. The list can be empty.
        definition: If provided, the index will be created with this custom index
                    definition.
        max_text_fields: If true, indexes will be encoded as if there were more than
                         32 text fields, allowing for additional fields beyond 32.
        temporary: Creates a lightweight temporary index which will expire after the
                   specified period of inactivity. The internal idle timer is reset
                   whenever the index is searched or added to.
        no_highlight: If true, disables highlighting support. Also implied by
                      `no_term_offsets`.
        no_term_frequencies: If true, term frequencies will not be saved in the
                             index.
        skip_initial_scan: If true, the initial scan and indexing will be skipped.

    """
    args = [CREATE_CMD, index_name]
    if definition is not None:
        args += definition.args
    if max_text_fields:
        args.append(MAXTEXTFIELDS)
    if temporary is not None and isinstance(temporary, int):
        args.append(TEMPORARY)
        args.append(str(temporary))
    if no_term_offsets:
        args.append(NOOFFSETS)
    if no_highlight:
        args.append(NOHL)
    if no_field_flags:
        args.append(NOFIELDS)
    if no_term_frequencies:
        args.append(NOFREQS)
    if skip_initial_scan:
        args.append(SKIPINITIALSCAN)
    if stopwords is not None and isinstance(stopwords, (list, tuple, set)):
        args += [STOPWORDS, str(len(stopwords))]
        if len(stopwords) > 0:
            args += list(stopwords)

    args.append("SCHEMA")
    try:
        args += list(itertools.chain(*(f.redis_args() for f in fields)))
    except TypeError:
        args += fields.redis_args()  # type: ignore

    default_node = client.get_default_node()
    return await default_node.execute_command(*args)


# TODO: The return type is incorrect because 5.x doesn't have "ProfileInformation"
def cluster_search(
    client: Search,
    query: Union[str, Query],
    query_params: Optional[Dict[str, Union[str, int, float, bytes]]] = None,
) -> Union[Result, Pipeline, Any]:  # type: ignore[type-arg]
    args, query = client._mk_query_args(query, query_params=query_params)
    st = time.monotonic()

    options = {}
    if get_protocol_version(client.client) not in ["3", 3]:
        options[NEVER_DECODE] = True

    node = client.client.get_default_node()
    res = client.execute_command(SEARCH_CMD, *args, **options, target_nodes=[node])

    if isinstance(res, Pipeline):
        return res

    return client._parse_results(
        SEARCH_CMD, res, query=query, duration=(time.monotonic() - st) * 1000.0
    )


# TODO: The return type is incorrect because 5.x doesn't have "ProfileInformation"
async def async_cluster_search(
    client: AsyncSearch,
    query: Union[str, Query],
    query_params: Optional[Dict[str, Union[str, int, float, bytes]]] = None,
) -> Union[Result, Pipeline, Any]:  # type: ignore[type-arg]
    args, query = client._mk_query_args(query, query_params=query_params)
    st = time.monotonic()

    options = {}
    if get_protocol_version(client.client) not in ["3", 3]:
        options[NEVER_DECODE] = True

    node = client.client.get_default_node()
    res = await client.execute_command(
        SEARCH_CMD, *args, **options, target_nodes=[node]
    )

    if isinstance(res, Pipeline):
        return res

    return client._parse_results(
        SEARCH_CMD, res, query=query, duration=(time.monotonic() - st) * 1000.0
    )


def _extract_hash_tag(key: str) -> str:
    """Extract hash tag from key. Returns empty string if no hash tag.

    Args:
        key (str): Redis key that may contain a hash tag.

    Returns:
        str: The hash tag including braces, or empty string if no hash tag.
    """
    start = key.find("{")
    if start == -1:
        return ""
    end = key.find("}", start + 1)
    if end == -1:
        return ""
    return key[start : end + 1]


def _keys_share_hash_tag(keys: List[str]) -> bool:
    """Check if all keys share the same hash tag for Redis Cluster compatibility.

    Args:
        keys (List[str]): List of Redis keys to check.

    Returns:
        bool: True if all keys share the same hash tag, False otherwise.
    """
    if not keys:
        return True

    first_tag = _extract_hash_tag(keys[0])
    return all(_extract_hash_tag(key) == first_tag for key in keys)


def is_cluster_url(url: str, **kwargs) -> bool:
    """
    Determine if the given URL and/or kwargs indicate a Redis Cluster connection.

    Args:
        url (str): The Redis connection URL.
        **kwargs: Additional keyword arguments that may indicate cluster usage.

    Returns:
        bool: True if the connection should be a cluster, False otherwise.
    """
    if "cluster" in kwargs and kwargs["cluster"]:
        return True
    if url:
        # Check if URL contains multiple hosts or has cluster flag
        if "," in url or "cluster=true" in url.lower():
            return True
    return False
