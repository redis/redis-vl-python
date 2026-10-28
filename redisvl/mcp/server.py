import asyncio
import logging
from contextlib import asynccontextmanager
from enum import Enum, auto
from importlib import import_module
from pathlib import Path
from typing import Any, Awaitable

from redis import __version__ as redis_py_version

from redisvl.exceptions import RedisSearchError, _is_missing_index_error
from redisvl.index import AsyncSearchIndex
from redisvl.mcp.auth import build_auth_provider, resolve_auth_config
from redisvl.mcp.config import MCPConfig, MCPIndexBindingConfig, load_mcp_config
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.runtime import BindingRuntime
from redisvl.mcp.settings import MCPSettings
from redisvl.mcp.tools.list_indexes import register_list_indexes_tool
from redisvl.mcp.tools.profiles import (
    register_profile_tools,
    validate_profile_against_schema,
)
from redisvl.mcp.tools.search import register_search_tool
from redisvl.mcp.tools.upsert import register_upsert_tool
from redisvl.mcp.transport_security import (
    HTTP_TRANSPORTS,
    build_host_origin_middleware,
    resolve_transport_security_config,
)
from redisvl.redis.connection import RedisConnectionFactory, is_version_gte
from redisvl.schema import IndexSchema

logger = logging.getLogger(__name__)


def _describe_scope(scope: frozenset[tuple[str, str]]) -> str:
    """Render one tool's injected scope for a startup error."""
    if not scope:
        return "injects nothing"
    return "injects " + ", ".join(
        f"{field} from claim '{claim}'" for field, claim in sorted(scope)
    )


def _describe_redis_index(redis_name: str, binding_ids: list[str]) -> str:
    """Render a Redis index and the bindings over it for a startup error."""
    noun = "bindings" if len(binding_ids) > 1 else "binding"
    return f"'{redis_name}' ({noun} {', '.join(repr(b) for b in sorted(binding_ids))})"


def _config_injects(config: Any) -> bool:
    """Report whether any configured profile injects a claim-derived filter."""
    if config is None:
        return False
    return any(profile.lock.inject for profile in config.custom_tools)


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
        self._registered_tool_fingerprint = ""
        self._registered_tools_inject = False
        # Set by run_async, the only point in the process that knows it. None
        # means an embedder started the server without naming a transport.
        self._transport: str | None = None

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

        # Host/Origin (DNS-rebinding) protection for HTTP transports. Resolved
        # here so config errors surface at construction; the bind-derived host
        # allowlist is finalized in run_async once host/port are known.
        self._transport_security = resolve_transport_security_config(
            settings, self._config_path
        )

        super().__init__("redisvl", lifespan=self._fastmcp_lifespan, auth=auth_provider)

    async def run_async(
        self,
        transport: Any = None,
        show_banner: bool | None = None,
        **transport_kwargs: Any,
    ) -> None:
        """Run the server, injecting Host/Origin validation for HTTP transports.

        The guard is prepended to any caller-supplied middleware so it runs
        outermost, rejecting DNS-rebinding requests before auth or tool handlers.
        ``stdio`` is untouched.
        """
        # Resolved the way FastMCP resolves it, because both the guard and the
        # tenant-injection check have to know what will actually be served. An
        # omitted transport falls back to `fastmcp.settings.transport`
        # (FASTMCP_TRANSPORT), which can name an HTTP transport; FastMCP before
        # 3.1 has no such setting and always served stdio, so that is the
        # fallback. Imported here because this module stays importable without
        # the `mcp` extra.
        import fastmcp

        settings = getattr(fastmcp, "settings", None)
        resolved = (
            transport
            if transport is not None
            else getattr(settings, "transport", "stdio")
        )
        self._transport = resolved

        if resolved in HTTP_TRANSPORTS:
            # The allowlist must match the address FastMCP binds, which falls
            # back to settings (FASTMCP_HOST / FASTMCP_PORT) the same way.
            # Hardcoded defaults would reject legitimate Host headers on a
            # settings-driven bind.
            host: Any = transport_kwargs.get("host")
            if host is None:
                host = getattr(settings, "host", "127.0.0.1")
            port: Any = transport_kwargs.get("port")
            if port is None:
                port = getattr(settings, "port", 8000)
            guard = build_host_origin_middleware(self._transport_security, host, port)
            if guard:
                existing = transport_kwargs.get("middleware") or []
                transport_kwargs["middleware"] = [*guard, *existing]

        await super().run_async(
            transport=transport, show_banner=show_banner, **transport_kwargs
        )

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

    def _validate_custom_tools_against_schema(self) -> None:
        """Fail startup on profiles that do not fit their bound index schema.

        Config load already checked naming, parameter policy, and that a pinned
        index exists. What it could not check is the schema, which is only known
        once the binding has been inspected -- so a locked projection or filter
        naming a missing field is caught here rather than silently matching
        nothing at request time.
        """
        config = getattr(self, "config", None)
        if config is None or not config.custom_tools:
            return

        for profile in config.custom_tools:
            binding_id = config.resolved_profile_index(profile)
            runtime = self._bindings[binding_id]
            validate_profile_against_schema(profile, runtime.schema)

    def _verify_injection_has_a_token(self, profiles: Any) -> None:
        """Refuse to start a claim-injection profile that can never succeed.

        Keyed off "auth is enabled", not off the transport name alone, so it
        covers an unauthenticated loopback bind and any --allow-unauthenticated
        bind as well as stdio. Without a verified token every call would be
        refused at request time -- safe, but advertised as a tool that works.

        Auth configured under stdio is the second case: FastMCP never
        authenticates stdio, so the verifier exists and is never consulted. Only
        a transport recorded by run_async counts; an embedder that never names
        one is left to the request-time refusal, which still fails closed.
        """
        injecting = sorted(profile.name for profile in profiles if profile.lock.inject)
        if not injecting:
            return

        names = ", ".join(injecting)
        if not self._auth_enabled:
            raise ValueError(
                f"custom_tools {names} inject a tenant filter from a token "
                "claim, but authentication is not enabled, so there is no "
                "verified token to read. Configure server.auth (or "
                "REDISVL_MCP_AUTH_*) and serve over an HTTP transport."
            )
        if self._transport == "stdio":
            raise ValueError(
                f"custom_tools {names} inject a tenant filter from a token "
                "claim, but the server is running over stdio, which is never "
                "authenticated. Serve over an HTTP transport (sse or "
                "streamable-http) so the configured auth applies."
            )

    def _verify_no_unscoped_route_to_injected_indexes(self, config: Any) -> None:
        """Refuse a tool surface that reaches a tenant-scoped index unscoped.

        Injection isolates a tool, but tenant data lives in an index, and every
        tool passes the same read scope gate. So any other route to that index
        -- the generic search, a write, or a custom tool injecting a different
        scope or none -- hands every caller the data the profile was meant to
        fence off, and a write can retag another tenant's document as the
        writer's own. That is configuration which voids the guarantee it
        declares, not a redundant surface, so it is refused rather than warned
        about.

        The scope is a property of the index, so every custom tool on it must
        inject the same entries. Injecting *something* is not enough: a second
        tool scoped by another field, or by the same field from another claim,
        reads across the tenants the first one separates.

        The index here is the Redis index, not the binding: two bindings with
        the same ``redis_name`` are two routes to one set of documents. The
        check sees only what this config names, so an alias, or a second index
        over the same key prefix, is outside it.
        """
        bindings: dict[str, list[str]] = {}
        for binding_id, binding in config.indexes.items():
            bindings.setdefault(binding.redis_name, []).append(binding_id)

        scopes: dict[str, dict[str, frozenset[tuple[str, str]]]] = {}
        for profile in config.custom_tools:
            binding = config.indexes[config.resolved_profile_index(profile)]
            scopes.setdefault(binding.redis_name, {})[profile.name] = frozenset(
                (entry.field, entry.claim) for entry in profile.lock.inject or ()
            )
        injected = sorted(
            redis_name
            for redis_name, by_tool in scopes.items()
            if any(by_tool.values())
        )
        if not injected:
            return

        problems: list[str] = []
        routes: list[str] = []
        if config.server.builtin_tool_enabled("search-records"):
            routes.append("search-records")
        if config.server.builtin_tool_enabled("upsert-records"):
            # Any writable binding over the index reaches its documents, not
            # only the ones the injecting tools are pinned to.
            writable = sorted(
                binding_id
                for redis_name in injected
                for binding_id in bindings[redis_name]
                if not (
                    self.mcp_settings.read_only or config.indexes[binding_id].read_only
                )
            )
            if writable:
                routes.append(
                    f"upsert-records via {', '.join(repr(b) for b in writable)}"
                )
        if routes:
            problems.append(
                "it is also reachable without that scope through "
                f"{', '.join(routes)}"
            )

        for redis_name in injected:
            by_tool = scopes[redis_name]
            if len(set(by_tool.values())) > 1:
                problems.append(
                    f"the custom tools on Redis index '{redis_name}' do not inject "
                    "the same scope: "
                    + "; ".join(
                        f"'{name}' {_describe_scope(scope)}"
                        for name, scope in sorted(by_tool.items())
                    )
                )
        if not problems:
            return

        described = ", ".join(
            _describe_redis_index(redis_name, bindings[redis_name])
            for redis_name in injected
        )
        raise ValueError(
            f"Redis index {described} is scoped by an injected token claim, but "
            f"{'; and '.join(problems)}. Each of those bypasses the tenant filter. "
            "Disable the built-ins with server.builtin_tools (for example "
            "'search-records: disabled'), mark every binding over the index "
            "read_only to stop writes, and give every custom tool on the index "
            "the same lock.inject. A tool meant to read across tenants belongs "
            "on a separate server with its own authentication."
        )

    @staticmethod
    def _tool_surface_fingerprint(config: Any) -> str:
        """Summarize the config that a registered tool set baked in."""
        if config is None:
            return ""
        return repr(
            (
                sorted(config.server.builtin_tools.items()),
                [profile.model_dump(mode="json") for profile in config.custom_tools],
            )
        )

    def _register_tools(self) -> None:
        """Register MCP tools once every binding is ready."""
        if self._tools_registered or not hasattr(self, "tool"):
            # Registration is deliberately once-per-process, since re-registering
            # the same names on the FastMCP object is not valid. Built-in closures
            # resolve their binding per call, so they survive a restart unchanged --
            # but *which* built-ins exist is a function of config, and a profile is
            # worse off still: its locked filter, projection, and signature are
            # fixed at registration. `startup()` re-reads the config file, so a
            # stop/start against an edited one keeps the old tool set either way.
            # The dangerous direction is an operator disabling a tool or tightening
            # a lock and believing the restart applied it.
            if self._tools_registered:
                config = getattr(self, "config", None)
                current = self._tool_surface_fingerprint(config)
                if current != self._registered_tool_fingerprint:
                    # For a tenant boundary "the old tools are still in effect"
                    # is not a log line. Either side counts: adding injection
                    # to a profile registered without it leaves that tool
                    # serving every tenant while the config says otherwise.
                    if self._registered_tools_inject or _config_injects(config):
                        raise RuntimeError(
                            "MCP tool configuration changed since tools were "
                            "registered, and claim injection is configured on "
                            "one side of the change. Tools register once per "
                            "process, so the previously registered tenant "
                            "scoping would stay in effect. Restart the process "
                            "to apply the new configuration."
                        )
                    logger.warning(
                        "MCP tool configuration (built-in or custom) changed "
                        "since tools were registered, but tools register once per "
                        "process. The previously registered tool set is still in "
                        "effect; "
                        "restart the process to apply the new configuration."
                    )
            return

        # The search description advertises schema-specific filter hints, which
        # are only unambiguous for a single binding. With multiple bindings the
        # caller selects an index per call, so fall back to the base description.
        search_schema: IndexSchema | None = None
        if len(self._bindings) == 1:
            search_schema = next(iter(self._bindings.values())).schema

        # An operator can disable a built-in whose curated profiles supersede it;
        # adding near-duplicate tools otherwise degrades the model's ability to
        # pick the right one.
        config = getattr(self, "config", None)
        enabled = (
            config.server.builtin_tool_enabled
            if config is not None
            else lambda _name: True
        )

        registered: list[str] = []

        # Discovery is on by default so clients can enumerate indexes; an
        # operator serving only curated profiles may still turn it off.
        discovery_enabled = enabled("list-indexes")
        if discovery_enabled:
            register_list_indexes_tool(self)
            registered.append("list-indexes")

        # `index` is required once several bindings exist, and without discovery
        # the logical ids cannot be learned any other way -- so every tool that
        # requires one has to name them inline instead of deferring to a tool that
        # is not published. Computed once so the two cannot drift apart.
        unlisted_index_ids = (
            sorted(self._bindings)
            if len(self._bindings) > 1 and not discovery_enabled
            else None
        )

        if enabled("search-records"):
            register_search_tool(self, search_schema, index_ids=unlisted_index_ids)
            registered.append("search-records")
        # Expose upsert only when at least one binding is writable. A binding is
        # read-only under global read-only mode or its own read_only policy, both
        # of which are folded into effective_read_only; the per-call write check
        # in the tool then rejects writes to any individual read-only binding.
        if enabled("upsert-records") and any(
            not rt.effective_read_only for rt in self._bindings.values()
        ):
            register_upsert_tool(self, index_ids=unlisted_index_ids)
            registered.append("upsert-records")
        registered.extend(register_profile_tools(self))

        self._warn_on_unusable_tool_surface(registered)
        self._registered_tool_fingerprint = self._tool_surface_fingerprint(config)
        self._registered_tools_inject = _config_injects(config)
        self._tools_registered = True

    def _warn_on_unusable_tool_surface(self, registered: list[str]) -> None:
        """Warn about tool-set shapes that are valid config but unusable in practice.

        Neither case is fatal -- an operator may be mid-rollout -- but both are
        silent otherwise, and both present to a client as a server that simply
        does not work.
        """
        if not registered:
            # Deliberately does not attribute a cause: `upsert-records` can also
            # be absent because every binding is read-only, not because
            # `builtin_tools` disabled it.
            logger.warning(
                "MCP server registered no tools, so clients will see an empty "
                "tool list. Check server.builtin_tools, custom_tools, and "
                "read-only settings."
            )
            return

        # Both `search-records` and `upsert-records` require an `index` once
        # several bindings exist, so either one is affected by losing discovery --
        # naming them in the descriptions keeps the contract satisfiable, but an
        # operator who disabled discovery on a multi-index server probably did not
        # intend to. Checking only search would leave a write-only surface silent.
        index_requiring = sorted(
            {"search-records", "upsert-records"}.intersection(registered)
        )
        if (
            len(self._bindings) > 1
            and index_requiring
            and "list-indexes" not in registered
        ):
            logger.warning(
                "MCP server has %d indexes and exposes %s, but list-indexes is "
                "disabled: clients cannot discover the logical index ids those "
                "tools require, so the ids are named inline in each tool "
                "description instead.",
                len(self._bindings),
                ", ".join(index_requiring),
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
        # Before any binding connects: both depend only on the loaded config,
        # so an unworkable one should not first need a reachable Redis.
        self._verify_injection_has_a_token(self.config.custom_tools)
        self._verify_no_unscoped_route_to_injected_indexes(self.config)
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
        # Validate before registering so a bad profile fails startup rather than
        # leaving a half-registered tool set behind.
        self._validate_custom_tools_against_schema()
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
            if _is_missing_index_error(exc):
                raise ValueError(
                    f"Configured Redis index '{binding.redis_name}' does not exist"
                ) from exc
            raise

        inspected_schema = binding.inspected_schema_from_index_info(index_info)
        return binding.to_index_schema(inspected_schema)

    @staticmethod
    def _make_index(schema: IndexSchema, client: Any) -> AsyncSearchIndex:
        """Bind an inspected schema and Redis client into an async index."""
        # The server acquired this client explicitly during startup, so hand
        # ownership to the index for a single shutdown path.
        return AsyncSearchIndex(schema=schema, redis_client=client, owns_client=True)

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
