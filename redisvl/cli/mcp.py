"""CLI entrypoint for the RedisVL MCP server."""

import argparse
import asyncio
import inspect
import sys


class _MCPArgumentParser(argparse.ArgumentParser):
    """ArgumentParser variant that reports usage errors with exit code 2."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, "%s: error: %s\n" % (self.prog, message))


class MCP:
    """Command handler for `rvl mcp`."""

    description = "Expose a configured Redis index to MCP clients for search and optional upsert operations."
    epilog = (
        "Use this command when wiring RedisVL into an MCP client.\n\n"
        "Examples:\n"
        "  rvl mcp --config /path/to/mcp_config.yaml\n"
        "  rvl mcp --config /path/to/mcp_config.yaml --transport streamable-http --port 8000\n"
        "  rvl mcp --config /path/to/mcp_config.yaml --transport sse --host 0.0.0.0 --port 9000"
    )
    usage = "\n".join(
        [
            "rvl mcp --config <path> [--read-only] [--transport <type>] [--host <host>] [--port <port>]\n",
            "\n",
        ]
    )

    def __init__(self):
        """Parse CLI arguments and run the MCP server command."""
        parser = _MCPArgumentParser(
            usage=self.usage,
            description=self.description,
            epilog=self.epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("--config", help="Path to MCP config file", required=True)
        parser.add_argument(
            "--read-only",
            help="Disable the upsert tool",
            action="store_true",
            dest="read_only",
            default=None,
        )
        parser.add_argument(
            "--transport",
            help="Transport protocol (default: stdio)",
            choices=["stdio", "sse", "streamable-http"],
            default="stdio",
        )
        parser.add_argument(
            "--host",
            help="Host to bind to for HTTP transports (default: 127.0.0.1)",
            default="127.0.0.1",
        )
        parser.add_argument(
            "--port",
            help="Port to bind to for HTTP transports (default: 8000)",
            type=int,
            default=8000,
        )
        parser.add_argument(
            "--allow-unauthenticated",
            help="Allow binding an HTTP transport to a non-loopback host without auth",
            action="store_true",
            dest="allow_unauthenticated",
            default=False,
        )

        args = parser.parse_args(sys.argv[2:])
        self._run(args)
        raise SystemExit(0)

    def _run(self, args):
        """Validate the environment, build the server, and serve stdio requests."""
        try:
            settings_cls, server_cls = self._load_mcp_components()
            settings = settings_cls.from_env(
                config=args.config,
                read_only=args.read_only,
            )
            server = server_cls(settings)
            self._run_awaitable(
                self._serve(
                    server,
                    transport=args.transport,
                    host=args.host,
                    port=args.port,
                    allow_unauthenticated=args.allow_unauthenticated,
                )
            )
        except KeyboardInterrupt:
            raise SystemExit(0)
        except Exception as exc:
            self._print_error(str(exc))
            raise SystemExit(1)

    @staticmethod
    def _load_mcp_components():
        """Import optional MCP dependencies only on the `rvl mcp` code path."""
        try:
            from redisvl.mcp import MCPSettings, RedisVLMCPServer
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "RedisVL MCP support requires optional dependencies. "
                "Install them with `pip install redisvl[mcp]`.\n"
                "Original error: %s" % exc
            )

        return MCPSettings, RedisVLMCPServer

    @staticmethod
    def _run_awaitable(awaitable):
        """Bridge the synchronous CLI entrypoint to async server lifecycle code."""
        return asyncio.run(awaitable)

    _LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

    @classmethod
    def _check_http_auth(cls, transport, host, auth_enabled, allow_unauthenticated):
        """Return a warning for an unauthenticated HTTP bind, or raise if unsafe.

        No-op for stdio or when auth is enabled. Warns for loopback binds and
        fails closed for non-loopback binds unless explicitly allowed.
        """
        if transport not in ("sse", "streamable-http") or auth_enabled:
            return None

        if host not in cls._LOOPBACK_HOSTS and not allow_unauthenticated:
            raise RuntimeError(
                f"Refusing to bind an unauthenticated MCP server to non-loopback "
                f"host '{host}'. Configure auth (REDISVL_MCP_AUTH_* or the "
                f"server.auth config block) or pass --allow-unauthenticated."
            )

        return (
            f"WARNING: serving MCP over {transport} on {host} without "
            f"authentication. Any client that can reach this address has full "
            f"access. Configure auth or restrict network access."
        )

    async def _serve(
        self,
        server,
        transport="stdio",
        host="127.0.0.1",
        port=8000,
        allow_unauthenticated=False,
    ):
        """Run startup, serving, and shutdown on one event loop."""
        warning = self._check_http_auth(
            transport,
            host,
            getattr(server, "_auth_enabled", False),
            allow_unauthenticated,
        )
        if warning:
            print(warning, file=sys.stderr)

        transport_kwargs = {}
        if transport in ("sse", "streamable-http"):
            transport_kwargs["host"] = host
            transport_kwargs["port"] = port

        # Prefer FastMCP's async transport path so it can own startup/shutdown
        # through its lifespan manager on the current event loop.
        run_async = getattr(server, "run_async", None)
        if callable(run_async):
            await run_async(transport=transport, **transport_kwargs)
            return

        started = False

        try:
            await server.startup()
            started = True

            result = server.run(transport=transport, **transport_kwargs)
            if inspect.isawaitable(result):
                await result

        finally:
            if started:
                try:
                    shutdown_result = server.shutdown()
                    if inspect.isawaitable(shutdown_result):
                        await shutdown_result
                except RuntimeError as exc:
                    # KeyboardInterrupt during stdio shutdown can leave FastMCP
                    # tearing down after the loop is already closing.
                    if "Event loop is closed" not in str(exc):
                        raise

    @staticmethod
    def _print_error(message):
        """Emit user-facing command errors to stderr."""
        print(message, file=sys.stderr)
