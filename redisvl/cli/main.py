import argparse
import sys

from redisvl.cli.index import Index
from redisvl.cli.stats import Stats
from redisvl.cli.version import Version
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


def _usage():
    usage = [
        "rvl <command> [<args>]\n",
        "Commands:",
        "\tindex       Index manipulation (create, delete, etc.)",
        "\tmcp         Run the RedisVL MCP server",
        "\tversion     Obtain the version of RedisVL",
        "\tstats       Obtain statistics about an index",
    ]
    return "\n".join(usage) + "\n"


class RedisVlCLI:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Redis Vector Library CLI", usage=_usage()
        )

        parser.add_argument("command", help="Subcommand to run")

        if len(sys.argv) < 2:
            parser.print_help(sys.stdout)
            sys.exit(0)

        args = parser.parse_args(sys.argv[1:2])
        
        if not hasattr(self, args.command):
            print(f"Unknown command: {args.command}\n", file=sys.stderr)
            parser.print_help(sys.stderr)
            sys.exit(2)
        
        try:
            getattr(self, args.command)()
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    def index(self):
        Index()
        sys.exit(0)

    def mcp(self):
        from redisvl.cli.mcp import MCP
        MCP()
        sys.exit(0)

    def version(self):
        Version()
        sys.exit(0)

    def stats(self):
        Stats()
        sys.exit(0)
