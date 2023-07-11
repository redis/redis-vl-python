import os
import sys
import argparse
from pprint import pprint
from typing import Any, List
from argparse import Namespace

from redisvl.index import SearchIndex
from redisvl.cli.utils import create_redis_url
from redisvl.utils.log import get_logger
from redisvl.utils.utils import convert_bytes
from redisvl.utils.connection import get_redis_connection

logger = get_logger(__name__)


class Index:

    usage = "\n".join([
        "redisvl index <command> [<args>]\n",
        "Commands:",
        "\tinfo        Obtain information about an index",
        "\tcreate      Create a new index",
        "\tdelete      Delete an existing index",
        "\tdestroy     Delete an existing index and all of its data",
        "\tlistall     List all indexes",
        "\n"
    ])

    def __init__(self):
        parser = argparse.ArgumentParser(usage=self.usage)

        parser.add_argument("command", help="Subcommand to run")

        parser.add_argument("-i", "--index", help="Index name", type=str, required=False)
        parser.add_argument(
            "-s", "--schema", help="Path to schema file", type=str, required=False
        )
        parser.add_argument("--host", help="Redis host", type=str, default="localhost")
        parser.add_argument("-p", "--port", help="Redis port", type=int, default=6379)
        parser.add_argument("--user", help="Redis username", type=str, default="default")
        parser.add_argument("--ssl", help="Use SSL", action="store_true")
        parser.add_argument(
            "-a", "--password", help="Redis password", type=str, default=""
        )

        args = parser.parse_args(sys.argv[2:])
        if not hasattr(self, args.command):
            parser.print_help()
            exit(0)
        try:
            getattr(self, args.command)(args)
        except Exception as e:
            logger.error(e)
            exit(0)


    def create(self, args: Namespace):
        """Create an index

        Usage:
            redisvl index create -i <index_name> | -s <schema_path>
        """
        if not args.schema:
            logger.error("Schema must be provided to create an index")
        index = SearchIndex.from_yaml(args.schema)
        url = create_redis_url(args)
        index.connect(url)
        index.create()
        logger.info("Index created successfully")

    def info(self, args: Namespace):
        """Obtain information about an index

        Usage:
            redisvl index info -i <index_name> | -s <schema_path>
        """
        index = self._connect_to_index(args)
        logger.info("Index information:")
        pprint(index.info())

    def listall(self, args: Namespace):
        """List all indices

        Usage:
            redisvl index listall
        """
        url = create_redis_url(args)
        conn = get_redis_connection(url)
        indices = convert_bytes(conn.execute_command("FT._LIST"))
        logger.info("Indices:")
        for i, index in enumerate(indices):
            logger.info(str(i+1) + ". " + index)

    def delete(self, args: Namespace, drop=False):
        """Delete an index

        Usage:
            redisvl index delete -i <index_name> | -s <schema_path>
        """
        index = self._connect_to_index(args)
        index.delete(drop=drop)
        logger.info("Index deleted successfully")

    def destroy(self, args: Namespace):
        """Delete an index and the documents within it

        Usage:
            redisvl index destroy -i <index_name> | -s <schema_path>
        """
        self.delete(args, drop=True)

    def _connect_to_index(self, args: Namespace) -> SearchIndex:

        # connect to redis
        try:
            url = create_redis_url(args)
            conn = get_redis_connection(url=url)
        except ValueError:
            logger.error("Must set REDIS_ADDRESS environment variable or provide host and port")
            exit(0)

        if args.index:
            index = SearchIndex.from_existing(conn, args.index)
        elif args.schema:
            index = SearchIndex.from_yaml(args.schema)
            index.set_client(conn)
        else:
            logger.error("Index name or schema must be provided")
            exit(0)

        return index
