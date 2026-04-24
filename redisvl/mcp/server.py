import asyncio
from contextlib import asynccontextmanager
from enum import Enum, auto
from importlib import import_module
from typing import Any, Awaitable

from redis import __version__ as redis_py_version

from redisvl.exceptions import RedisSearchError
from redisvl.index import AsyncSearchIndex
from redisvl.mcp.config import MCPConfig, load_mcp_config
from redisvl.mcp.settings import MCPSettings
from redisvl.mcp.tools.search import register_search_tool
from redisvl.mcp.tools.upsert import register_upsert_tool
from redisvl.redis.connection import RedisConnectionFactory, is_version_gte
from redisvl.schema import IndexSchema

try:
    from fastmcp import FastMCP
except ImportError:

    class FastMCP:  # type: ignore[no-redef]
        """Import-safe stand-in used when the optional MCP SDK is unavailable."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs


def resolve_vectorizer_class(class_name: str) -> type[Any]:
    """Resolve a vectorizer class from the public RedisVL vectorizer module."""
    vectorize_module = import_module("redisvl.utils.vectorize")
    try:
        return getattr(vectorize_module, class_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown vectorizer class: {class_name}") from exc


class _LifecycleState(Enum):
    INITIAL = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()


class RedisVLMCPServer(FastMCP):
    """MCP server exposing RedisVL capabilities for one existing Redis index."""

    _LifecycleState = _LifecycleState

    def __init__(self, settings: MCPSettings):
        """Create a server shell with lazy config, index, and vectorizer state."""
        self.mcp_settings = settings
        self.config: MCPConfig | None = None
        self._index: AsyncSearchIndex | None = None
        self._vectorizer: Any | None = None
        self._supports_native_hybrid_search: bool | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._tools_registered = False

        # Lifecycle management
        self._lifecycle_state = _LifecycleState.INITIAL  # Server lifecycle
        self._transition_lock = asyncio.Lock()  # Prevents overlapping startup/shutdown
        self._request_state_lock = asyncio.Lock()  # Guards request admission state
        self._active_requests = 0
        self._active_requests_drained = (
            asyncio.Event()
        )  # Set when no requests are active
        self._active_requests_drained.set()
        self._fastmcp_lifespan = self._server_lifespan  # FastMCP startup/shutdown hook

        super().__init__("redisvl", lifespan=self._fastmcp_lifespan)

    async def startup(self) -> None:
        """Load config, inspect the configured index, and initialize dependencies."""
        async with self._transition_lock:
            await self._begin_startup()
            client = None
            try:
                client = await self._initialize_runtime_resources()
                await self._mark_running()
            except Exception:
                await self._teardown_runtime(client)
                await self._mark_stopped()
                raise

    async def shutdown(self) -> None:
        """Release owned vectorizer and Redis resources."""
        async with self._transition_lock:
            if await self._begin_shutdown():
                # _begin_shutdown() returns True when startup never finished or teardown already ran.
                return

            await self._wait_for_active_requests()
            try:
                await self._teardown_runtime()
            finally:
                await self._mark_stopped()

    async def get_index(self) -> AsyncSearchIndex:
        """Return the initialized async index or fail if startup has not run."""
        if self._index is None:
            raise RuntimeError("MCP server has not been started")
        return self._index

    async def get_vectorizer(self) -> Any:
        """Return the initialized vectorizer or fail if startup has not run."""
        if self.config is None:
            raise RuntimeError("MCP server has not been started")
        if self._vectorizer is None:
            raise RuntimeError("MCP server vectorizer is not configured")
        return self._vectorizer

    async def run_guarded(self, operation_name: str, awaitable: Awaitable[Any]) -> Any:
        """Run a coroutine under the configured concurrency and timeout limits."""
        del operation_name
        semaphore = self._semaphore
        if semaphore is None:
            self._close_awaitable(awaitable)
            raise RuntimeError("MCP server is not running")

        async with semaphore:
            async with self._request_state_lock:
                if self._lifecycle_state is not _LifecycleState.RUNNING:
                    self._close_awaitable(awaitable)
                    raise RuntimeError("MCP server is not running")

                config = self.config
                if config is None:
                    self._close_awaitable(awaitable)
                    raise RuntimeError("MCP server is not running")

                self._active_requests += 1
                self._active_requests_drained.clear()

            try:
                return await asyncio.wait_for(
                    awaitable,
                    timeout=config.runtime.request_timeout_seconds,
                )
            finally:
                async with self._request_state_lock:
                    self._active_requests -= 1
                    if self._active_requests == 0:
                        self._active_requests_drained.set()

    def _build_vectorizer(self) -> Any:
        """Instantiate the configured vectorizer class from validated config."""
        if self.config is None:
            raise RuntimeError("MCP server config not loaded")
        if self.config.vectorizer is None:
            raise RuntimeError("MCP server vectorizer is not configured")

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

    async def supports_native_hybrid_search(self) -> bool:
        """Return whether the current runtime supports Redis native hybrid search."""
        if self._supports_native_hybrid_search is not None:
            return self._supports_native_hybrid_search
        if self._index is None:
            raise RuntimeError("MCP server has not been started")
        if not is_version_gte(redis_py_version, "7.1.0"):
            self._supports_native_hybrid_search = False
            return False

        client = await self._index._get_client()
        info = await client.info("server")
        if not is_version_gte(info.get("redis_version", "0.0.0"), "8.4.0"):
            self._supports_native_hybrid_search = False
            return False

        self._supports_native_hybrid_search = hasattr(
            client.ft(self._index.schema.index.name), "hybrid_search"
        )
        return self._supports_native_hybrid_search

    def _register_tools(self) -> None:
        """Register MCP tools once the server is ready."""
        if self._tools_registered or not hasattr(self, "tool"):
            return

        register_search_tool(self)
        if not self.mcp_settings.read_only:
            register_upsert_tool(self)
        self._tools_registered = True

    @staticmethod
    def _is_missing_index_error(exc: RedisSearchError) -> bool:
        """Detect the Redis search errors that mean the configured index is absent."""
        message = str(exc).lower()
        return "unknown index name" in message or "no such index" in message

    @asynccontextmanager
    async def _server_lifespan(self, _server: Any):
        """Bridge FastMCP lifespan hooks onto the server's explicit lifecycle."""
        await self.startup()
        try:
            yield {}
        finally:
            await self.shutdown()

    async def _teardown_runtime(self, client: Any | None = None) -> None:
        """Release runtime resources and clear terminal state."""
        vectorizer = self._vectorizer
        index = self._index
        self._vectorizer = None
        self._index = None
        self.config = None
        self._semaphore = None

        try:
            if vectorizer is not None:
                aclose = getattr(vectorizer, "aclose", None)
                close = getattr(vectorizer, "close", None)
                if callable(aclose):
                    await aclose()
                elif callable(close):
                    close()
        finally:
            self._supports_native_hybrid_search = None
            if index is not None:
                await index.disconnect()
            elif client is not None:
                await client.aclose()

    @staticmethod
    def _close_awaitable(awaitable: Awaitable[Any]) -> None:
        """Close coroutine objects we reject before awaiting to avoid warnings."""
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()

    async def _begin_startup(self) -> None:
        """Move the server into STARTING or fail on invalid transitions."""
        async with self._request_state_lock:
            if self._lifecycle_state in (
                _LifecycleState.STARTING,
                _LifecycleState.RUNNING,
            ):
                raise RuntimeError("MCP server is already running")
            self._lifecycle_state = _LifecycleState.STARTING

    async def _mark_running(self) -> None:
        """Mark the server as fully initialized and ready to admit requests."""
        async with self._request_state_lock:
            self._lifecycle_state = _LifecycleState.RUNNING

    async def _begin_shutdown(self) -> bool:
        """Move the server into STOPPING unless it is already stopped."""
        async with self._request_state_lock:
            if self._lifecycle_state in (
                _LifecycleState.INITIAL,
                _LifecycleState.STOPPED,
            ):
                self.config = None
                self._semaphore = None
                self._lifecycle_state = _LifecycleState.STOPPED
                return True

            self._lifecycle_state = _LifecycleState.STOPPING
            return False

    async def _mark_stopped(self) -> None:
        """Mark the server as fully stopped."""
        async with self._request_state_lock:
            self._lifecycle_state = _LifecycleState.STOPPED

    async def _wait_for_active_requests(self) -> None:
        """Wait for already-admitted guarded requests to finish."""
        await self._active_requests_drained.wait()

    async def _initialize_runtime_resources(self) -> Any:
        """Load config and initialize the Redis-backed runtime dependencies."""
        self.config = load_mcp_config(self.mcp_settings.config)
        self._semaphore = asyncio.Semaphore(self.config.runtime.max_concurrency)
        self._supports_native_hybrid_search = None
        timeout = self.config.runtime.startup_timeout_seconds

        client = await self._connect_redis_client(timeout)
        try:
            effective_schema = await self._load_effective_schema(client, timeout)
            self._initialize_index(effective_schema, client)
            self.config.validate_search(
                schema=effective_schema,
                supports_native_hybrid_search=await self.supports_native_hybrid_search(),
            )
            if self.config.requires_startup_vectorizer:
                await self._initialize_vectorizer(effective_schema, timeout)
            self._register_tools()
            return client
        except Exception:
            if self._index is None:
                await client.aclose()
            raise

    async def _connect_redis_client(self, timeout: int) -> Any:
        """Connect to Redis and verify the server is reachable."""
        if self.config is None:
            raise RuntimeError("MCP server config not loaded")

        client = await asyncio.wait_for(
            RedisConnectionFactory._get_aredis_connection(
                redis_url=self.config.server.redis_url
            ),
            timeout=timeout,
        )
        await asyncio.wait_for(client.info("server"), timeout=timeout)
        return client

    async def _load_effective_schema(self, client: Any, timeout: int) -> IndexSchema:
        """Inspect the configured Redis index and build the effective schema."""
        if self.config is None:
            raise RuntimeError("MCP server config not loaded")

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
        return self.config.to_index_schema(inspected_schema)

    def _initialize_index(self, schema: IndexSchema, client: Any) -> None:
        """Bind the inspected schema and Redis client into an async index."""
        self._index = AsyncSearchIndex(schema=schema, redis_client=client)
        # The server acquired this client explicitly during startup, so hand
        # ownership to the index for a single shutdown path.
        self._index._owns_redis_client = True

    async def _initialize_vectorizer(self, schema: IndexSchema, timeout: int) -> None:
        """Build the configured vectorizer and validate it against the schema."""
        self._vectorizer = await asyncio.wait_for(
            asyncio.to_thread(self._build_vectorizer),
            timeout=timeout,
        )
        self._validate_vectorizer_dims(schema)
