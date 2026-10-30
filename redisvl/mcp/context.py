"""The handle a code tool gets on Redis.

A code tool never receives the server, its ``BindingRuntime``, the real
``AsyncSearchIndex`` or a Redis client. It receives a :class:`ToolContext`,
whose ``index`` is an allow-listed wrapper: every method on it is spelled out
here, routes through ``server.run_guarded`` (the process-wide concurrency cap
and the binding's request timeout), and write methods exist only on the
handle given to a tool that declared it writes. A method added to
``AsyncSearchIndex`` later is therefore unreachable until it is added here.

Every method stays inside the bound index: searches and aggregations run
against its name, and records are addressed by document id, which the index
turns into a key under its own prefix. Nothing accepts a full Redis key or a
query that names its own target, because either could reach another index's
documents, including a tenant-scoped one. An index whose prefix overlaps
another's shares keys with it, which no handle can tell apart; an index with
an empty prefix covers every key, so its document ids are full Redis keys.

This is a guardrail, not a sandbox: Python can still reach private attributes.
It makes the safe path the only obvious one.
"""

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from redisvl.index.index import DEFAULT_BULK_BATCH_SIZE
from redisvl.mcp.tools.search import _embed_query
from redisvl.query.sql import SQLQuery

if TYPE_CHECKING:
    from redis.commands.search.aggregation import AggregateResult
    from redis.commands.search.result import Result

    from redisvl.index import AsyncSearchIndex
    from redisvl.mcp.runtime import BindingRuntime
    from redisvl.schema import IndexSchema


def _refuse_sql(query: Any) -> None:
    if isinstance(query, SQLQuery):
        raise ValueError(
            "a code tool cannot run a SQLQuery: it names its own index, so it "
            "could read beyond the one the tool is bound to"
        )


class IndexHandle:
    """Read access to one bound index, bounded like every built-in tool call."""

    __slots__ = ("_server", "_index", "_timeout_seconds")

    def __init__(
        self, server: Any, index: "AsyncSearchIndex", *, timeout_seconds: float
    ) -> None:
        self._server = server
        self._index = index
        self._timeout_seconds = timeout_seconds

    async def _guarded(self, operation_name: str, awaitable: Any) -> Any:
        return await self._server.run_guarded(
            operation_name, awaitable, timeout_seconds=self._timeout_seconds
        )

    async def query(self, query: Any) -> Any:
        """Run a RedisVL query object and return its processed results.

        A ``CountQuery`` returns the match count rather than records. A
        ``SQLQuery`` is refused: its ``FROM`` clause, not the bound index,
        decides which index it reads.
        """
        _refuse_sql(query)
        return await self._guarded("query", self._index.query(query))

    async def batch_query(
        self, queries: list[Any], batch_size: int = 10
    ) -> list[list[dict[str, Any]]]:
        """Run several RedisVL query objects and return each one's results."""
        for query in queries:
            _refuse_sql(query)
        return await self._guarded(
            "batch_query", self._index.batch_query(queries, batch_size=batch_size)
        )

    async def search(self, *args: Any, **kwargs: Any) -> "Result":
        """Run a raw ``FT.SEARCH`` and return the redis-py result."""
        return await self._guarded("search", self._index.search(*args, **kwargs))

    async def batch_search(
        self, queries: list[Any], batch_size: int = 10
    ) -> list["Result"]:
        """Run several raw searches and return each redis-py result."""
        return await self._guarded(
            "batch_search", self._index.batch_search(queries, batch_size=batch_size)
        )

    async def aggregate(self, *args: Any, **kwargs: Any) -> "AggregateResult":
        """Run a raw ``FT.AGGREGATE`` and return the redis-py result."""
        return await self._guarded("aggregate", self._index.aggregate(*args, **kwargs))

    async def fetch(self, id: str) -> dict[str, Any] | None:
        """Return one record by its id, or ``None`` when it does not exist."""
        return await self._guarded("fetch", self._index.fetch(id))

    async def exists(self) -> bool:
        """Return whether the index exists in Redis."""
        return await self._guarded("exists", self._index.exists())

    async def info(self) -> dict[str, Any]:
        """Return ``FT.INFO`` for the bound index."""
        return await self._guarded("info", self._index.info())

    def document_id(self, key: str) -> str:
        """Return the document id of a Redis key under this index's prefix.

        Search results carry each record's full key as ``id``, while the
        handle addresses records by document id, so this is the bridge from a
        result back to ``fetch`` or a write. Raises ``ValueError`` for a key
        outside the prefix.
        """
        prefix = self._index.key("")
        if not prefix:
            return key
        if not key.startswith(prefix) or key == prefix:
            raise ValueError(f"key {key!r} is not a document of this index")
        return key[len(prefix) :]


class WritableIndexHandle(IndexHandle):
    """An :class:`IndexHandle` that may also write, for tools declared as writing.

    Only writes whose cost is bounded by the input they are given are exposed;
    the filter-wide bulk operations are not, because the request timeout can
    cancel them partway through.
    """

    __slots__ = ()

    async def load(
        self,
        data: Iterable[Any],
        id_field: str | None = None,
        ttl: int | None = None,
        preprocess: Callable | None = None,
        batch_size: int | None = None,
    ) -> list[str]:
        """Write records to the index and return their document ids.

        Each record's id is its ``id_field`` value, or a generated one when
        ``id_field`` is not given. The ids, not full Redis keys, are what
        ``fetch``, ``drop_documents`` and ``expire_documents`` take.
        """
        keys = await self._guarded(
            "load",
            self._index.load(
                data,
                id_field=id_field,
                ttl=ttl,
                preprocess=preprocess,
                batch_size=batch_size,
            ),
        )
        return [self.document_id(key) for key in keys]

    async def drop_documents(
        self, ids: str | list[str], batch_size: int = DEFAULT_BULK_BATCH_SIZE
    ) -> int:
        """Delete records by document id and return how many were removed."""
        return await self._guarded(
            "drop_documents", self._index.drop_documents(ids, batch_size=batch_size)
        )

    async def expire_documents(self, ids: str | list[str], ttl: int) -> int | list[int]:
        """Set a time-to-live, in seconds, on records by document id."""
        keys = (
            self._index.key(ids)
            if isinstance(ids, str)
            else [self._index.key(id) for id in ids]
        )
        return await self._guarded(
            "expire_documents", self._index.expire_keys(keys, ttl)
        )


def _freeze(value: Any) -> Any:
    """Return a read-only copy: mappings become proxies, lists become tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _current_claims() -> Mapping[str, Any] | None:
    """The verified claims of the current request, or ``None`` without a token."""
    # Imported here because this module stays importable without the `mcp` extra.
    from fastmcp.server.dependencies import get_access_token

    token = get_access_token()
    if token is None:
        return None
    return _freeze(getattr(token, "claims", None) or {})


class ToolContext:
    """Everything a code tool may use to reach Redis, and nothing else.

    ``index`` is the bound index's handle. ``schema`` is a copy of the
    binding's schema, and ``vectorizer`` is the binding's. ``claims`` is a
    read-only view of the current request's verified token claims, or ``None``
    when the request carries no token.
    """

    __slots__ = ("_index", "_schema", "_vectorizer", "_claims")

    def __init__(
        self,
        *,
        index: IndexHandle,
        schema: "IndexSchema",
        vectorizer: Any | None,
        claims: Mapping[str, Any] | None,
    ) -> None:
        self._index = index
        self._schema = schema
        self._vectorizer = vectorizer
        self._claims = claims

    @classmethod
    def for_request(
        cls, server: Any, runtime: "BindingRuntime", *, writes: bool
    ) -> "ToolContext":
        """Build the context for one call of a tool bound to ``runtime``."""
        handle_type = WritableIndexHandle if writes else IndexHandle
        return cls(
            index=handle_type(
                server,
                runtime.index,
                timeout_seconds=runtime.binding.runtime.request_timeout_seconds,
            ),
            # A copy, because the index holds the same object: a tool that
            # edited it would retarget every later call on the binding,
            # built-in tools included.
            schema=runtime.schema.model_copy(deep=True),
            vectorizer=runtime.vectorizer,
            claims=_current_claims(),
        )

    @property
    def index(self) -> IndexHandle:
        return self._index

    @property
    def schema(self) -> "IndexSchema":
        return self._schema

    @property
    def vectorizer(self) -> Any | None:
        return self._vectorizer

    @property
    def claims(self) -> Mapping[str, Any] | None:
        return self._claims

    async def embed(self, text: str) -> Any:
        """Embed ``text`` with the binding's vectorizer."""
        if self._vectorizer is None:
            raise RuntimeError("MCP server vectorizer is not configured")
        return await _embed_query(self._vectorizer, text)
