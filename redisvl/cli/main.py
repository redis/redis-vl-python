import argparse
import sys

from redisvl.utils.log import get_logger

logger = get_logger(__name__)


def _usage():
    return "rvl <command> [<args>]"


def _command_overview():
    command_groups = [
        "Command groups:",
        "  index       Create, inspect, list, and delete Redis search indexes",
        "  stats       Show statistics for an existing Redis search index",
        "  migrate     Plan, apply, and validate index migrations (experimental)",
        "  version     Show the installed RedisVL version",
        "  mcp         Run the RedisVL MCP server",
    ]
    return "\n".join(command_groups)


def _examples():
    examples = [
        "Examples:",
        "  rvl index --help",
        "  rvl index create -s schema.yaml",
        "  rvl stats -i user_index",
        "  rvl mcp --config /path/to/mcp.yaml",
    ]
    return "\n".join(examples)


class RedisVlCLI:
    def __init__(self):
        parser = argparse.ArgumentParser(
            prog="rvl",
            description=f"Redis Vector Library CLI.\n\n{_command_overview()}",
            usage=_usage(),
            epilog=_examples(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument("command", nargs="?", help="Command group to run")

        if len(sys.argv) < 2:
            parser.print_help(sys.stdout)
            sys.exit(0)

        args = parser.parse_args(sys.argv[1:2])

        if not args.command or not hasattr(self, args.command):
            print(f"Unknown command: {args.command}\n", file=sys.stderr)
            parser.print_help(sys.stderr)
            sys.exit(2)

        try:
            getattr(self, args.command)()
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    def index(self):
        from redisvl.cli.index import Index

        Index()
        sys.exit(0)

    def mcp(self):
        from redisvl.cli.mcp import MCP

        MCP()
        sys.exit(0)

    def version(self):
        from redisvl.cli.version import Version

        Version()
        sys.exit(0)

    def migrate(self):
        from redisvl.cli.migrate import Migrate

        Migrate()
        sys.exit(0)

    def stats(self):
        from redisvl.cli.stats import Stats

        Stats()
        sys.exit(0)
