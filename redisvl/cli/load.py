import sys
import argparse
import asyncio
import typing as t

from redisvl.index import SearchIndex
from redisvl.load import concurrent_store_as_hash
from redisvl.readers import pandas
from redisvl.schema import read_schema
from redisvl.utils.connection import get_async_redis_connection

from redisvl.utils.log import get_logger
logger = get_logger(__name__)


class Load:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Load vector data into redis"
        )
        parser.add_argument("-d", "--data", help="Path to data file", type=str, required=True)
        parser.add_argument("-s", "--schema", help="Path to schema file", type=str, required=True)
        parser.add_argument("--host", help="Redis host", type=str, default="localhost")
        parser.add_argument("-p", "--port", help="Redis port", type=int, default=6379)
        parser.add_argument("-a", "--password", help="Redis password", type=str, default="")
        parser.add_argument("-c", "--concurrency", type=int, default=50)
        parser.add_argument("-v", "--vector", help="Vector field name in data", type=str, default="vector")
        # TODO add argument to optionally not create index
        args = parser.parse_args(sys.argv[2:])

        # TODO more data types
        # TODO try/expect and clear data upon failure
        asyncio.run(self.load_pandas(args))

    async def load_pandas(self, args):

        # Create Redis Connection
        logger.info(f"Connecting to {args.host}:{str(args.port)}")
        redis_conn = get_async_redis_connection(args.host, args.port, args.password)
        logger.info("Connected.")

        # Create index
        logger.info("Validating data schema...")
        index_attrs, fields = read_schema(args.schema)
        index = SearchIndex(redis_conn, **index_attrs)
        logger.info("Data schema validated.")

        # read data
        logger.info("Reading data...")
        data = pandas.from_pickle(args.data)
        logger.info("Data read.")

        # load data
        logger.info("Loading data...")
        await concurrent_store_as_hash(data, args.concurrency, args.vector, index.prefix, redis_conn)
        logger.info("Data loaded.")

        logger.info("Creating index...")
        await index.create(fields)
        logger.info("Index created.")