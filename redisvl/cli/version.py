import argparse
import sys
from argparse import Namespace

from redisvl import __version__
from redisvl.utils.log import get_logger

logger = get_logger("[RedisVL]")


class Version:
    usage = "\n".join(
        [
            "rvl version [<args>]\n",
            "\n",
        ]
    )

    def __init__(self):
        parser = argparse.ArgumentParser(usage=self.usage)
        parser.add_argument(
            "-s", "--short", help="Print only the version number", action="store_true"
        )

        args = parser.parse_args(sys.argv[2:])
        self.version(args)

    def version(self, args: Namespace):
        if args.short:
            print(__version__)
        else:
            logger.info(f"RedisVL version {__version__}")
