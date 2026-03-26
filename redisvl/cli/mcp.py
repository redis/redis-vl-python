"""CLI entrypoint for the RedisVL MCP server."""

import argparse
import asyncio
import inspect
import sys


class _MCPArgumentParser(argparse.ArgumentParser):
    """ArgumentParser variant that reports usage errors with exit code 1."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, "%s: error: %s\n" % (self.prog, message))


class MCP:
    """Command handler for `rvl mcp`."""

    description = "Expose a configured Redis index to MCP clients for search and optional upsert operations."
    epilog = (
        "Use this command when wiring RedisVL into an MCP client.\n\n"
        "Example:\n"
        "  uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp_config.yaml"
    )
    usage = "\n".join(
        [
            "rvl mcp --config <path> [--read-only]\n",
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

        args = parser.parse_args(sys.argv[2:])
        self._run(args)
        raise SystemExit(0)

    def _run(self, args):
        """Validate the environment, build the server, and serve stdio requests."""
        try:
            self._ensure_supported_python()
            settings_cls, server_cls = self._load_mcp_components()
            settings = settings_cls.from_env(
                config=args.config,
                read_only=args.read_only,
            )
            server = server_cls(settings)
            self._run_awaitable(self._serve(server))
        except KeyboardInterrupt:
            raise SystemExit(0)
        except Exception as exc:
            self._print_error(str(exc))
            raise SystemExit(1)

    @staticmethod
    def _ensure_supported_python():
        """Fail fast when the current interpreter cannot support MCP extras."""
        if sys.version_info < (3, 10):
            version = "%s.%s.%s" % (
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            )
            raise RuntimeError(
                "RedisVL MCP CLI requires Python 3.10 or newer. "
                "Current runtime is Python %s." % version
            )

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

    async def _serve(self, server):
        """Run startup, stdio serving, and shutdown on one event loop."""
        started = False

        try:
            await server.startup()
            started = True

            # Prefer FastMCP's async transport path so startup, serving, and
            # shutdown all share the same event loop.
            run_async = getattr(server, "run_async", None)
            if callable(run_async):
                await run_async(transport="stdio")
            else:
                result = server.run(transport="stdio")
                if inspect.isawaitable(result):
                    await result
        finally:
            if started:
                try:
                    result = server.shutdown()
                    if inspect.isawaitable(result):
                        await result
                except RuntimeError as exc:
                    # KeyboardInterrupt during stdio shutdown can leave FastMCP
                    # tearing down after the loop is already closing.
                    if "Event loop is closed" not in str(exc):
                        raise

    @staticmethod
    def _print_error(message):
        """Emit user-facing command errors to stderr."""
        print(message, file=sys.stderr)
