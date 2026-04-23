import os
import sys
from argparse import ArgumentParser, Namespace
from urllib.parse import quote, urlparse, urlunparse

import yaml
from pydantic import ValidationError

from redisvl.exceptions import RedisSearchError
from redisvl.index import SearchIndex
from redisvl.redis.constants import REDIS_URL_ENV_VAR
from redisvl.utils.log import get_logger

logger = get_logger("[RedisVL]")
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379


def _has_explicit_connection_options(args: Namespace) -> bool:
    return any(
        (
            getattr(args, "host", None) is not None,
            getattr(args, "port", None) is not None,
            bool(getattr(args, "user", None)),
            bool(getattr(args, "password", None)),
        )
    )


def _get_auth_credentials(args: Namespace) -> tuple[str | None, str | None]:
    return getattr(args, "user", None) or None, getattr(args, "password", None) or None


def _build_redis_url(args: Namespace) -> str:
    scheme = "rediss" if getattr(args, "ssl", False) else "redis"
    host = getattr(args, "host", None)
    if host is None:
        host = DEFAULT_REDIS_HOST

    port = getattr(args, "port", None)
    if port is None:
        port = DEFAULT_REDIS_PORT
    user, password = _get_auth_credentials(args)

    auth = ""
    if user:
        auth = quote(user, safe="")
        if password:
            auth += f":{quote(password, safe='')}"
        auth += "@"
    elif password:
        auth = f":{quote(password, safe='')}@"

    return f"{scheme}://{auth}{host}:{port}"


def _apply_ssl_scheme(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.scheme == "rediss":
        return url
    return urlunparse(parsed_url._replace(scheme="rediss"))


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
        if getattr(args, "ssl", False):
            return _apply_ssl_scheme(env_address)
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
        "-a",
        "--password",
        help="Redis password",
        type=str,
        default=None,
    )
    return parser


# Exceptions commonly raised when loading or validating a schema path (-s).
SCHEMA_INPUT_ERRORS = (
    FileNotFoundError,
    ValueError,
    yaml.YAMLError,
    ValidationError,
)


def cli_index_target_name(args: Namespace, index: SearchIndex | None) -> str:
    if index is not None:
        return index.schema.index.name
    return args.index or args.schema or "unknown"


def exit_schema_input_error(args: Namespace, exc: BaseException) -> None:
    if not args.schema:
        raise exc
    print(f"Invalid schema file or path ({args.schema!r}): {exc}", file=sys.stderr)
    sys.exit(2)


def exit_redis_search_error(
    args: Namespace, index: SearchIndex | None, exc: RedisSearchError
) -> None:
    name = cli_index_target_name(args, index)
    print(
        f"Redis search operation failed for index {name!r}: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)
