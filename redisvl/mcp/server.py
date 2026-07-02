import asyncio
import logging
from contextlib import asynccontextmanager
from enum import Enum, auto
from importlib import import_module
from pathlib import Path
from typing import Any, Awaitable

from redis import __version__ as redis_py_version

from redisvl.exceptions import RedisSearchError
from redisvl.index import AsyncSearchIndex
from redisvl.mcp.auth import build_auth_provider, resolve_auth_config
from redisvl.mcp.config import MCPConfig, MCPIndexBindingConfig, load_mcp_config
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.runtime import BindingRuntime
from redisvl.mcp.settings import MCPSettings
from redisvl.mcp.tools.list_indexes import register_list_indexes_tool
from redisvl.mcp.tools.search import register_search_tool
from redisvl.mcp.tools.upsert import register_upsert_tool
from redisvl.redis.connection import RedisConnectionFactory, is_version_gte
from redisvl.schema import IndexSchema

logger = logging.getLogger(__name__)

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
    """MCP server exposing RedisVL capabilities for one or many existing indexes."""

    _LifecycleState = _LifecycleState

    def __init__(self, settings: MCPSettings):
        """Create a server shell with lazy config and per-binding runtime state."""
        self.mcp_settings = settings
        self.config: MCPConfig | None = None
        self._bindings: dict[str, BindingRuntime] = {}
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

        # Resolve the config path to an absolute path so a later working-directory
        # change cannot make the construction-time and startup-time reads diverge.
        self._config_path = str(Path(settings.config).expanduser().resolve())

        # Auth is resolved at construction time (FastMCP needs the provider in
        # its constructor), reading env vars and peeking the YAML server.auth
        # block without running full startup. Applies only to HTTP transports.
        auth_config = resolve_auth_config(settings, self._config_path)
        auth_provider = build_auth_provider(auth_config)
        self.auth_config = auth_config
        self._auth_enabled = auth_provider is not None

        super().__init__("redisvl", lifespan=self._fastmcp_lifespan, auth=auth_provider)

    async def startup(self) -> None:
        """Load config, inspect the configured index, and initialize dependencies."""
        async with self._transition_lock:
            await self._begin_startup()
            try:
                await self._initialize_runtime_resources()
                await self._mark_running()
            except Exception:
                # Fail closed: release whatever initialization built before
                # marking the server stopped. This is the single teardown path
                # for any post-begin failure -- binding init, tool registration,
                # or a later step such as _mark_running -- so resources are
                # never leaked regardless of where startup fails.
                await self._teardown_runtime()
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

    def resolve_binding(self, index_id: str | None) -> BindingRuntime:
        """Resolve the runtime for a logical index id, honoring single-index defaults.

        - ``None`` with exactly one configured binding returns that binding,
          preserving backward-compatible single-index behavior.
        - ``None`` with multiple bindings is an ``invalid_request``; the caller
          must name an index.
        - An unknown id is an ``invalid_request``.

        Write-availability is not enforced here; that is the upsert tool's job.
        """
        if not self._bindings:
            raise RuntimeError("MCP server has not been started")

        if index_id is None:
            if len(self._bindings) == 1:
                return next(iter(self._bindings.values()))
            available = ", ".join(sorted(self._bindings))
            raise RedisVLMCPError(
                "index is required when multiple indexes are configured; "
                f"available: {available}",
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )

        runtime = self._bindings.get(index_id)
        if runtime is None:
            available = ", ".join(sorted(self._bindings))
            raise RedisVLMCPError(
                f"Unknown index '{index_id}'; available: {available}",
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )
        return runtime

    async def run_guarded(
        self,
        operation_name: str,
        awaitable: Awaitable[Any],
        *,
        timeout_seconds: float,
    ) -> Any:
        """Run a coroutine under the global concurrency cap and a request timeout.

        The timeout is sourced per-binding by the caller; the concurrency
        semaphore is a single process-wide ceiling shared across all bindings.
        """
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

                if self.config is None:
                    self._close_awaitable(awaitable)
                    raise RuntimeError("MCP server is not running")

                self._active_requests += 1
                self._active_requests_drained.clear()

            try:
                return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
            finally:
                async with self._request_state_lock:
                    self._active_requests -= 1
                    if self._active_requests == 0:
                        self._active_requests_drained.set()

    @staticmethod
    def _build_vectorizer(binding: MCPIndexBindingConfig) -> Any:
        """Instantiate a binding's configured vectorizer class from its config."""
        if binding.vectorizer is None:
            raise RuntimeError("MCP server vectorizer is not configured")

        vectorizer_class = resolve_vectorizer_class(binding.vectorizer.class_name)
        return vectorizer_class(**binding.vectorizer.to_init_kwargs())

    @staticmethod
    def _validate_vectorizer_dims(
        binding: MCPIndexBindingConfig, vectorizer: Any, schema: IndexSchema
    ) -> None:
        """Fail startup when vectorizer dimensions disagree with schema dimensions."""
        if vectorizer is None:
            return

        configured_dims = binding.get_vector_field_dims(schema)
        actual_dims = getattr(vectorizer, "dims", None)
        if (
            configured_dims is not None
            and actual_dims is not None
            and configured_dims != actual_dims
        ):
            raise ValueError(
                f"Vectorizer dims {actual_dims} do not match configured vector field dims {configured_dims}"
            )

    @staticmethod
    async def _probe_native_hybrid_search(index: AsyncSearchIndex) -> bool:
        """Probe whether a connected index supports Redis native hybrid search."""
        if not is_version_gte(redis_py_version, "7.1.0"):
            return False

        client = await index._get_client()
        info = await client.info("server")
        if not is_version_gte(info.get("redis_version", "0.0.0"), "8.4.0"):
            return False

        return hasattr(client.ft(index.schema.index.name), "hybrid_search")

    def _register_tools(self) -> None:
        """Register MCP tools once every binding is ready."""
        if self._tools_registered or not hasattr(self, "tool"):
            return

        # The search description advertises schema-specific filter hints, which
        # are only unambiguous for a single binding. With multiple bindings the
        # caller selects an index per call, so fall back to the base description.
        search_schema: IndexSchema | None = None
        if len(self._bindings) == 1:
            search_schema = next(iter(self._bindings.values())).schema

        # Discovery is always available so clients can enumerate indexes.
        register_list_indexes_tool(self)
        register_search_tool(self, search_schema)
        # Expose upsert only when at least one binding is writable. A binding is
        # read-only under global read-only mode or its own read_only policy, both
        # of which are folded into effective_read_only; the per-call write check
        # in the tool then rejects writes to any individual read-only binding.
        if any(not rt.effective_read_only for rt in self._bindings.values()):
            register_upsert_tool(self)
        self._tools_registered = True

    @staticmethod
    def _is_missing_index_error(exc: RedisSearchError) -> bool:
        """Detect the Redis search errors that mean the configured index is absent.

        Different RediSearch versions phrase the error differently
        (``unknown index name``, ``no such index``, or ``SEARCH_INDEX_NOT_FOUND
        Index not found``), so check for each known wording.
        """
        message = str(exc).lower()
        return (
            "unknown index name" in message
            or "no such index" in message
            or "search_index_not_found" in message
            or "index not found" in message
        )

    @asynccontextmanager
    async def _server_lifespan(self, _server: Any):
        """Bridge FastMCP lifespan hooks onto the server's explicit lifecycle."""
        await self.startup()
        try:
            yield {}
        finally:
            await self.shutdown()

    @staticmethod
    async def _close_resources(
        *, index: Any | None, vectorizer: Any | None, client: Any | None = None
    ) -> None:
        """Close one binding's vectorizer and Redis connection.

        A fully built binding owns its client through ``index``; a binding that
        failed mid-startup may have a bare ``client`` and no index yet.
        """
        try:
            if vectorizer is not None:
                aclose = getattr(vectorizer, "aclose", None)
                close = getattr(vectorizer, "close", None)
                if callable(aclose):
                    await aclose()
                elif callable(close):
                    close()
        finally:
            if index is not None:
                await index.disconnect()
            elif client is not None:
                await client.aclose()

    async def _teardown_runtime(self) -> None:
        """Release every binding's runtime resources and clear terminal state.

        ``_tools_registered`` is intentionally *not* reset here: MCP tools are
        registered once on the FastMCP instance and their closures resolve the
        live binding at call time, so they survive teardown and remain valid
        across a stop/start. Resetting it would make a restart re-register the
        same tool names on the instance.
        """
        bindings = list(self._bindings.values())
        self._bindings = {}
        self.config = None
        self._semaphore = None

        for runtime in bindings:
            try:
                await self._close_resources(
                    index=runtime.index, vectorizer=runtime.vectorizer
                )
            except Exception:
                logger.warning(
                    "error closing binding %s during teardown",
                    runtime.binding_id,
                    exc_info=True,
                )

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
                self._bindings = {}
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

    def _verify_auth_not_stale(self) -> None:
        """Fail closed if startup-time auth disagrees with what was wired.

        The auth provider must be passed to FastMCP at construction, before the
        full config is loaded. If the config file was unreadable then (for
        example created after construction), auth could be silently disabled
        while the loaded config enables it. Refuse to serve rather than expose
        an unauthenticated HTTP transport.
        """
        expected = resolve_auth_config(self.mcp_settings, self._config_path)
        if (expected is not None) != self._auth_enabled:
            raise RuntimeError(
                "MCP auth configuration changed between server construction and "
                "startup, so the wired auth state is stale. Refusing to start to "
                "avoid serving unauthenticated. Use an absolute config path and "
                "ensure the config file exists before constructing the server."
            )

    async def _initialize_runtime_resources(self) -> None:
        """Load config and initialize every configured binding independently."""
        self.config = load_mcp_config(self._config_path)
        self._verify_auth_not_stale()
        # The semaphore is a single process-wide concurrency ceiling shared by
        # all bindings; take the max across bindings. This means the most
        # permissive binding sets the cap — e.g. five bindings each configured
        # with max_concurrency=2 yield Semaphore(2), not Semaphore(10).
        self._semaphore = asyncio.Semaphore(
            max(
                binding.runtime.max_concurrency
                for binding in self.config.indexes.values()
            )
        )
        self._bindings = {}

        # On failure, startup()'s handler tears down any bindings built here, so
        # this method does not need its own teardown. (A binding that fails
        # mid-build closes its own bare client inside _initialize_binding.)
        for binding_id, binding in self.config.indexes.items():
            self._bindings[binding_id] = await self._initialize_binding(
                binding_id, binding
            )
        self._register_tools()

    async def _initialize_binding(
        self, binding_id: str, binding: MCPIndexBindingConfig
    ) -> BindingRuntime:
        """Inspect, validate, and initialize a single configured binding."""
        timeout = binding.runtime.startup_timeout_seconds
        client = await self._connect_redis_client(timeout)
        index: AsyncSearchIndex | None = None
        vectorizer: Any | None = None
        try:
            schema = await self._load_effective_schema(binding, client, timeout)
            index = self._make_index(schema, client)
            supports_native_hybrid = await self._probe_native_hybrid_search(index)
            binding.validate_search(
                schema=schema,
                supports_native_hybrid_search=supports_native_hybrid,
            )
            if binding.requires_startup_vectorizer:
                vectorizer = await self._initialize_vectorizer(binding, schema, timeout)
            return BindingRuntime(
                binding_id=binding_id,
                binding=binding,
                index=index,
                schema=schema,
                vectorizer=vectorizer,
                supports_native_hybrid_search=supports_native_hybrid,
                effective_read_only=self.mcp_settings.read_only or binding.read_only,
            )
        except Exception:
            await self._close_resources(
                index=index, vectorizer=vectorizer, client=client
            )
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

    async def _load_effective_schema(
        self, binding: MCPIndexBindingConfig, client: Any, timeout: int
    ) -> IndexSchema:
        """Inspect a binding's Redis index and build its effective schema."""
        try:
            index_info = await asyncio.wait_for(
                AsyncSearchIndex._info(binding.redis_name, client),
                timeout=timeout,
            )
        except RedisSearchError as exc:
            if self._is_missing_index_error(exc):
                raise ValueError(
                    f"Configured Redis index '{binding.redis_name}' does not exist"
                ) from exc
            raise

        inspected_schema = binding.inspected_schema_from_index_info(index_info)
        return binding.to_index_schema(inspected_schema)

    @staticmethod
    def _make_index(schema: IndexSchema, client: Any) -> AsyncSearchIndex:
        """Bind an inspected schema and Redis client into an async index."""
        index = AsyncSearchIndex(schema=schema, redis_client=client)
        # The server acquired this client explicitly during startup, so hand
        # ownership to the index for a single shutdown path.
        index._owns_redis_client = True
        return index

    async def _initialize_vectorizer(
        self, binding: MCPIndexBindingConfig, schema: IndexSchema, timeout: int
    ) -> Any:
        """Build a binding's vectorizer and validate it against the schema."""
        vectorizer = await asyncio.wait_for(
            asyncio.to_thread(self._build_vectorizer, binding),
            timeout=timeout,
        )
        self._validate_vectorizer_dims(binding, vectorizer, schema)
        return vectorizer
