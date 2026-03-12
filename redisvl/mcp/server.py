import asyncio
from importlib import import_module
from typing import Any, Awaitable, Optional, Type

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.config import MCPConfig, load_mcp_config
from redisvl.mcp.settings import MCPSettings

try:
    from mcp.server.fastmcp import FastMCP
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
    """MCP server exposing RedisVL vector search capabilities.

    This server manages the lifecycle of a Redis vector index and an embedding
    vectorizer, providing Model Context Protocol (MCP) tools for semantic search
    operations. It handles configuration loading, connection management,
    concurrency limits, and graceful shutdown of resources.
    """

    def __init__(self, settings: MCPSettings):
        """Create a server shell with lazy config, index, and vectorizer state."""
        super().__init__("redisvl")
        self.mcp_settings = settings
        self.config: Optional[MCPConfig] = None
        self._index: Optional[AsyncSearchIndex] = None
        self._vectorizer: Optional[Any] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def startup(self) -> None:
        """Load config, validate Redis/index state, and initialize dependencies."""
        self.config = load_mcp_config(self.mcp_settings.config)
        self._semaphore = asyncio.Semaphore(self.config.runtime.max_concurrency)
        self._index = AsyncSearchIndex(
            schema=self.config.to_index_schema(),
            redis_url=self.config.redis_url,
        )
        try:
            timeout = self.config.runtime.startup_timeout_seconds
            index_exists = await asyncio.wait_for(self._index.exists(), timeout=timeout)
            if not index_exists:
                if self.config.runtime.index_mode == "validate_only":
                    raise ValueError(
                        f"Index '{self.config.index.name}' does not exist for validate_only mode"
                    )
                await asyncio.wait_for(self._index.create(), timeout=timeout)

            # Vectorizer construction may perform provider-specific setup, so keep it
            # off the event loop and bound it with the same startup timeout.
            self._vectorizer = await asyncio.wait_for(
                asyncio.to_thread(self._build_vectorizer),
                timeout=timeout,
            )
            self._validate_vectorizer_dims()
        except Exception:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Release owned vectorizer and Redis resources."""
        try:
            if self._vectorizer is not None:
                aclose = getattr(self._vectorizer, "aclose", None)
                close = getattr(self._vectorizer, "close", None)
                if callable(aclose):
                    await aclose()
                elif callable(close):
                    close()
                self._vectorizer = None
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

        # The semaphore centralizes backpressure so future tool handlers do not
        # each need to reimplement request-limiting behavior.
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

    def _validate_vectorizer_dims(self) -> None:
        """Fail startup when vectorizer dimensions disagree with schema dimensions."""
        if self.config is None or self._vectorizer is None:
            return

        configured_dims = self.config.vector_field_dims
        actual_dims = getattr(self._vectorizer, "dims", None)
        if (
            configured_dims is not None
            and actual_dims is not None
            and configured_dims != actual_dims
        ):
            raise ValueError(
                f"Vectorizer dims {actual_dims} do not match configured vector field dims {configured_dims}"
            )
