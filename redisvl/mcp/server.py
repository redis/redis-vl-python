import asyncio
from importlib import import_module
from typing import Any, Awaitable, Optional, Type

from redisvl.exceptions import RedisSearchError
from redisvl.index import AsyncSearchIndex
from redisvl.mcp.config import MCPConfig, load_mcp_config
from redisvl.mcp.settings import MCPSettings
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.schema import IndexSchema

try:
    from fastmcp import FastMCP
except ImportError:

    class FastMCP:  # type: ignore[no-redef]
        """Import-safe stand-in used when the optional MCP SDK is unavailable."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs


def resolve_vectorizer_class(class_name: str) -> Type[Any]:
    """Resolve a vectorizer class from the public RedisVL vectorizer module."""
    vectorize_module = import_module("redisvl.utils.vectorize")
    try:
        return getattr(vectorize_module, class_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown vectorizer class: {class_name}") from exc


class RedisVLMCPServer(FastMCP):
    """MCP server exposing RedisVL capabilities for one existing Redis index."""

    def __init__(self, settings: MCPSettings):
        """Create a server shell with lazy config, index, and vectorizer state."""
        super().__init__("redisvl")
        self.mcp_settings = settings
        self.config: Optional[MCPConfig] = None
        self._index: Optional[AsyncSearchIndex] = None
        self._vectorizer: Optional[Any] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def startup(self) -> None:
        """Load config, inspect the configured index, and initialize dependencies."""
        self.config = load_mcp_config(self.mcp_settings.config)
        self._semaphore = asyncio.Semaphore(self.config.runtime.max_concurrency)
        timeout = self.config.runtime.startup_timeout_seconds
        client = None

        try:
            client = await asyncio.wait_for(
                RedisConnectionFactory._get_aredis_connection(
                    redis_url=self.config.server.redis_url
                ),
                timeout=timeout,
            )
            await asyncio.wait_for(client.info("server"), timeout=timeout)

            try:
                index_info = await asyncio.wait_for(
                    AsyncSearchIndex._info(self.config.redis_name, client),
                    timeout=timeout,
                )
            except RedisSearchError as exc:
                if self._is_missing_index_error(exc):
                    raise ValueError(
                        f"Configured Redis index '{self.config.redis_name}' does not exist"
                    ) from exc
                raise

            inspected_schema = self.config.inspected_schema_from_index_info(index_info)
            effective_schema = self.config.to_index_schema(inspected_schema)
            self._index = AsyncSearchIndex(schema=effective_schema, redis_client=client)
            # The server acquired this client explicitly during startup, so hand
            # ownership to the index for a single shutdown path.
            self._index._owns_redis_client = True

            self._vectorizer = await asyncio.wait_for(
                asyncio.to_thread(self._build_vectorizer),
                timeout=timeout,
            )
            self._validate_vectorizer_dims(effective_schema)
        except Exception:
            if self._index is not None:
                await self.shutdown()
            elif client is not None:
                await client.aclose()
            raise

    async def shutdown(self) -> None:
        """Release owned vectorizer and Redis resources."""
        vectorizer = self._vectorizer
        self._vectorizer = None
        try:
            if vectorizer is not None:
                aclose = getattr(vectorizer, "aclose", None)
                close = getattr(vectorizer, "close", None)
                if callable(aclose):
                    await aclose()
                elif callable(close):
                    close()
        finally:
            if self._index is not None:
                index = self._index
                self._index = None
                await index.disconnect()

    async def get_index(self) -> AsyncSearchIndex:
        """Return the initialized async index or fail if startup has not run."""
        if self._index is None:
            raise RuntimeError("MCP server has not been started")
        return self._index

    async def get_vectorizer(self) -> Any:
        """Return the initialized vectorizer or fail if startup has not run."""
        if self._vectorizer is None:
            raise RuntimeError("MCP server has not been started")
        return self._vectorizer

    async def run_guarded(self, operation_name: str, awaitable: Awaitable[Any]) -> Any:
        """Run a coroutine under the configured concurrency and timeout limits."""
        del operation_name
        if self.config is None or self._semaphore is None:
            raise RuntimeError("MCP server has not been started")

        async with self._semaphore:
            return await asyncio.wait_for(
                awaitable,
                timeout=self.config.runtime.request_timeout_seconds,
            )

    def _build_vectorizer(self) -> Any:
        """Instantiate the configured vectorizer class from validated config."""
        if self.config is None:
            raise RuntimeError("MCP server config not loaded")

        vectorizer_class = resolve_vectorizer_class(self.config.vectorizer.class_name)
        return vectorizer_class(**self.config.vectorizer.to_init_kwargs())

    def _validate_vectorizer_dims(self, schema: IndexSchema) -> None:
        """Fail startup when vectorizer dimensions disagree with schema dimensions."""
        if self.config is None or self._vectorizer is None:
            return

        configured_dims = self.config.get_vector_field_dims(schema)
        actual_dims = getattr(self._vectorizer, "dims", None)
        if (
            configured_dims is not None
            and actual_dims is not None
            and configured_dims != actual_dims
        ):
            raise ValueError(
                f"Vectorizer dims {actual_dims} do not match configured vector field dims {configured_dims}"
            )

    @staticmethod
    def _is_missing_index_error(exc: RedisSearchError) -> bool:
        """Detect the Redis search errors that mean the configured index is absent."""
        message = str(exc).lower()
        return "unknown index name" in message or "no such index" in message
