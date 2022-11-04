import asyncio
import warnings
import argparse
import logging
import pickle
import typing as t
import numpy as np

from redis.asyncio import Redis
from data.schema import get_schema
from utils.search_index import SearchIndex


warnings.filterwarnings("error")

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)5s:%(filename)25s"
    ":%(lineno)3s %(funcName)30s(): %(message)s",
)

def read_data(data_file: str) -> t.List[dict]:
    """
    Read dataset from a pickled dataframe (Pandas) file.
    TODO -- add support for other input data types.

    Args:
        data_file (str): Path to the destination
                         of the input data file.

    Returns:
        t.List[dict]: List of Hash objects to insert to Redis.
    """
    logging.info(f"Reading dataset from file: {data_file}")
    with open(data_file, "rb") as f:
        df = pickle.load(f)
    return df.to_dict("records")

async def gather_with_concurrency(
    *data,
    n: int,
    vector_field_name: str,
    prefix: str,
    redis_conn: Redis
):
    """
    Gather and load the hashes into Redis using
    async connections.

    Args:
        n (int): Max number of "concurrent" async connections.
        vector_field_name (str): Vector field name in the dataframe.
        prefix (str): Redis key prefix for all hashes in the search index.
        redis_conn (Redis): Redis connection.
    """
    logging.info("Loading dataset into Redis")
    semaphore = asyncio.Semaphore(n)
    async def load(d: dict):
        async with semaphore:
            d[vector_field_name] = np.array(d[vector_field_name], dtype = np.float32).tobytes()
            key = prefix + str(d["id"])
            await redis_conn.hset(key, mapping = d)
    # gather with concurrency
    await asyncio.gather(*[load(d) for d in data])

async def load_all_data(
    redis_conn: Redis,
    concurrency: int,
    prefix: str,
    vector_field_name: str,
    data_file: str,
    index_name: str
):
    """
    Load all data.

    Args:
        redis_conn (Redis): Redis connection.
        concurrency (int): Max number of "concurrent" async connections.
        prefix (str): Redis key prefix for all hashes in the search index.
        vector_field_name (str): Vector field name in the dataframe.
        data_file (str): Path to the destination of the input data file.
        index_name (str): Name of the RediSearch Index.
    """
    search_index = SearchIndex(
        index_name = index_name,
        redis_conn = redis_conn
    )

    # Load from pickled dataframe file
    data = read_data(data_file)

    # Gather async
    await gather_with_concurrency(
        *data,
        n = concurrency,
        prefix = prefix,
        vector_field_name= vector_field_name,
        redis_conn = redis_conn
    )

    # Load schema
    logging.info("Processing RediSearch schema")
    schema = get_schema(len(data))
    await search_index.create(*schema, prefix=prefix)
    logging.info("All done. Data uploaded and RediSearch index created.")


async def main():
    # Parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Redis host", type=str, default="localhost")
    parser.add_argument("-p", "--port", help="Redis port", type=int, default=6379)
    parser.add_argument("-a", "--password", help="Redis password", type=str, default="")
    parser.add_argument("-c", "--concurrency", type=int, default=50)
    parser.add_argument("-d", "--data", help="Path to data file", type=str, default="data/embeddings.pkl")
    parser.add_argument("--prefix", help="Key prefix for all hashes in the search index", type=str, default="vector:")
    parser.add_argument("-v", "--vector", help="Vector field name in df", type=str, default="vector")
    parser.add_argument("-i", "--index", help="Index name", type=str, default="index")
    args = parser.parse_args()

    # Create Redis Connection
    connection_args = {
        "host": args.host,
        "port": args.port
    }
    if args.password:
        connection_args.update({"password": args.password})
    redis_conn = Redis(**connection_args)

    # Perform data loading
    await load_all_data(
        redis_conn=redis_conn,
        concurrency=args.concurrency,
        prefix=args.prefix,
        vector_field_name=args.vector,
        data_file=args.data,
        index_name=args.index
    )


if __name__ == "__main__":
    asyncio.run(main())