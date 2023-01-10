import sys
import argparse
import typing as t
from pprint import pprint

from redisvl.index import SearchIndex
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


    def create(self, args):
        """Create an index

        Usage:
            redisvl index create -i <index_name> | -s <schema_path>
        """
        if not args.schema:
            logger.error("Schema must be provided to create an index")
        index = SearchIndex.from_yaml(args.schema)
        index.connect(host=args.host,
                      port=args.port,
                      username=args.user,
                      password=args.password)
        index.create()
        logger.info("Index created successfully")

    def info(self, args):
        """Obtain information about an index

        Usage:
            redisvl index info -i <index_name> | -s <schema_path>
        """
        index = self._connect_to_index(args)
        logger.info(pprint(index.info()))

    def listall(self, args):
        """List all indices

        Usage:
            redisvl index listall
        """
        conn = get_redis_connection(
            host=args.host,
            port=args.port,
            username=args.user,
            password=args.password
        )
        indices = convert_bytes(conn.execute_command("FT._LIST"))
        logger.info("Indices:")
        for i, index in enumerate(indices):
            logger.info(str(i+1) + ". " + index)

    def delete(self, args, drop=False):
        """Delete an index

        Usage:
            redisvl index delete -i <index_name> | -s <schema_path>
        """
        index = self._connect_to_index(args)
        index.delete(drop=drop)
        logger.info("Index deleted successfully")

    def destroy(self, args):
        """Delete an index and the documents within it

        Usage:
            redisvl index destroy -i <index_name> | -s <schema_path>
        """
        self.delete(args, drop=True)

    def _connect_to_index(self, args: t.Any) -> SearchIndex:
        if args.index:
            index = SearchIndex.from_existing(args.index)
        elif args.schema:
            index = SearchIndex.from_yaml(args.schema)
        else:
            logger.error("Index name or schema must be provided")
            exit(0)

        index.connect(
            host=args.host,
            port=args.port,
            username=args.user,
            password=args.password
        )
        return index