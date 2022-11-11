import argparse
import asyncio
import sys
import typing as t

from redisvl import readers
from redisvl.index import SearchIndex
from redisvl.load import concurrent_store_as_hash
from redisvl.utils.connection import get_async_redis_connection
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


class Load:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Load vector data into redis")
        parser.add_argument(
            "-d", "--data", help="Path to data file", type=str, required=True
        )
        parser.add_argument(
            "-s", "--schema", help="Path to schema file", type=str, required=True
        )
        parser.add_argument("--host", help="Redis host", type=str, default="localhost")
        parser.add_argument("-p", "--port", help="Redis port", type=int, default=6379)
        parser.add_argument(
            "-a", "--password", help="Redis password", type=str, default=""
        )
        parser.add_argument("-c", "--concurrency", type=int, default=50)
        # TODO add argument to optionally not create index
        args = parser.parse_args(sys.argv[2:])
        if not args.data:
            parser.print_help()
            exit(0)

        # Create Redis Connection
        try:
            logger.info(f"Connecting to {args.host}:{str(args.port)}")
            redis_conn = get_async_redis_connection(args.host, args.port, args.password)
            logger.info("Connected.")
        except:
            # TODO: be more specific about the exception
            logger.error("Could not connect to redis.")
            exit(1)

        # validate schema
        index = SearchIndex.from_yaml(redis_conn, args.schema)

        # read in data
        logger.info("Reading data...")
        data = self.read_data(args)  # TODO add other readers and formats
        logger.info("Data read.")

        # load data and create the index
        asyncio.run(self.load_and_create_index(args.concurrency, data, index))

    def read_data(
        self, args: t.List[str], reader: str = "pandas", format: str = "pickle"
    ) -> dict:
        if reader == "pandas":
            if format == "pickle":
                return readers.pandas.from_pickle(args.data)
            else:
                raise NotImplementedError(
                    "Only pickle format is supported for pandas reader."
                )
        else:
            raise NotImplementedError("Only pandas reader is supported.")

    async def load_and_create_index(
        self, concurrency: int, data: dict, index: SearchIndex
    ):

        logger.info("Loading data...")
        if index.storage_type == "hash":
            await concurrent_store_as_hash(
                data, concurrency, index.key_field, index.prefix, index.redis_conn
            )
        else:
            raise NotImplementedError("Only hash storage type is supported.")
        logger.info("Data loaded.")

        # create index
        logger.info("Creating index...")
        await index.create()
        logger.info("Index created.")
