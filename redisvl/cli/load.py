import argparse
import asyncio
import sys
import typing as t

from redisvl.index import AsyncSearchIndex
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
        parser.add_argument("-r", "--reader", help="Reader", type=str, default="pandas")
        parser.add_argument("-f", "--format", help="Format", type=str, default="pickle")
        parser.add_argument("-c", "--concurrency", type=int, default=50)
        # TODO add argument to optionally not create index
        args = parser.parse_args(sys.argv[2:])
        if not args.data:
            parser.print_help()
            exit(0)

        # validate schema
        index = AsyncSearchIndex.from_yaml(args.schema)

        # try to connect to redis
        index.connect(host=args.host, port=args.port, password=args.password)

        # read in data
        logger.info("Reading data...")
        reader = self._get_reader(args)
        logger.info("Data read.")

        # load data and create the index
        asyncio.run(self._load_and_create_index(args.concurrency, reader, index))

    def _get_reader(self, args: t.List[str]) -> dict:
        if args.reader == "pandas":
            from redisvl.readers import PandasReader

            if args.format == "pickle":
                return PandasReader.from_pickle(args.data)
            elif args.format == "json":
                return PandasReader.from_json(args.data)
            else:
                raise NotImplementedError(
                    "Only pickle and json formats are supported for pandas reader using the CLI"
                )
        else:
            raise NotImplementedError("Only pandas reader is supported.")

    async def _load_and_create_index(
        self, concurrency: int, reader: t.Iterable[dict], index: AsyncSearchIndex
    ):

        logger.info("Loading data...")
        await index.load(data=reader, concurrency=concurrency)
        logger.info("Data loaded.")

        # create index
        logger.info("Creating index...")
        await index.create()
        logger.info("Index created.")
