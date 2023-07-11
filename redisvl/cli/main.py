import argparse
import sys

from redisvl.cli.index import Index
from redisvl.cli.load import Load
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


def _usage():
    usage = [
        "redisvl <command> [<args>]\n",
        "Commands:",
        "\tload        Load vector data into redis",
        "\tindex       Index manipulation (create, delete, etc.)",
        "\tquery       Query an existing index",
    ]
    return "\n".join(usage)


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

    def load(self):
        Load()
        exit(0)

    def index(self):
        Index()
        exit(0)
