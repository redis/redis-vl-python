import os
from argparse import ArgumentParser, Namespace

from redisvl.utils.log import get_logger

logger = get_logger("[RedisVL]")


def create_redis_url(args: Namespace) -> str:
    env_address = os.getenv("REDIS_URL")
    if env_address:
        logger.info(f"Using Redis address from environment variable, REDIS_URL")
        return env_address
    elif args.url:
        return args.url
    else:
        url = "redis://"
        if args.ssl:
            url += "rediss://"
        if args.user:
            url += args.user
            if args.password:
                url += ":" + args.password
            url += "@"
        url += args.host + ":" + str(args.port)
        return url


def add_index_parsing_options(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("-i", "--index", help="Index name", type=str, required=False)
    parser.add_argument(
        "-s", "--schema", help="Path to schema file", type=str, required=False
    )
    parser.add_argument("-u", "--url", help="Redis URL", type=str, required=False)
    parser.add_argument("--host", help="Redis host", type=str, default="localhost")
    parser.add_argument("-p", "--port", help="Redis port", type=int, default=6379)
    parser.add_argument("--user", help="Redis username", type=str, default="default")
    parser.add_argument("--ssl", help="Use SSL", action="store_true")
    parser.add_argument("-a", "--password", help="Redis password", type=str, default="")
    return parser
