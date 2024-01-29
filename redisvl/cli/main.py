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
            parser.print_help()
            exit(0)

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            parser.print_help()
            exit(0)
        getattr(self, args.command)()

    def index(self):
        Index()
        exit(0)

    def version(self):
        Version()
        exit(0)

    def stats(self):
        Stats()
        exit(0)
