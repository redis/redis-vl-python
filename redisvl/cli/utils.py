import os
from argparse import ArgumentParser, Namespace

from redisvl.redis.constants import REDIS_URL_ENV_VAR
from redisvl.utils.log import get_logger

logger = get_logger("[RedisVL]")
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379


def _has_explicit_connection_options(args: Namespace) -> bool:
    return any(
        getattr(args, attribute, None) is not None
        for attribute in ("host", "port", "user", "password")
    ) or bool(getattr(args, "ssl", False))


def _build_redis_url(args: Namespace) -> str:
    scheme = "rediss" if getattr(args, "ssl", False) else "redis"
    host = getattr(args, "host", None) or DEFAULT_REDIS_HOST
    port = getattr(args, "port", None) or DEFAULT_REDIS_PORT
    user = getattr(args, "user", None)
    password = getattr(args, "password", None)

    auth = ""
    if user:
        auth = user
        if password:
            auth += f":{password}"
        auth += "@"
    elif password:
        auth = f":{password}@"

    return f"{scheme}://{auth}{host}:{port}"


def create_redis_url(args: Namespace) -> str:
    if args.url:
        return args.url
    if _has_explicit_connection_options(args):
        return _build_redis_url(args)

    env_address = os.getenv(REDIS_URL_ENV_VAR)
    if env_address:
        logger.info(
            f"Using Redis address from environment variable, {REDIS_URL_ENV_VAR}"
        )
        return env_address

    return _build_redis_url(args)


def add_index_parsing_options(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("-i", "--index", help="Index name", type=str, required=False)
    parser.add_argument(
        "-s", "--schema", help="Path to schema file", type=str, required=False
    )
    parser.add_argument("-u", "--url", help="Redis URL", type=str, required=False)
    parser.add_argument("--host", help="Redis host", type=str, default=None)
    parser.add_argument("-p", "--port", help="Redis port", type=int, default=None)
    parser.add_argument("--user", help="Redis username", type=str, default=None)
    parser.add_argument("--ssl", help="Use SSL", action="store_true")
    parser.add_argument(
        "-a", "--password", help="Redis password", type=str, default=None
    )
    return parser
