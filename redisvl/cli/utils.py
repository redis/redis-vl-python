import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any, Mapping
from urllib.parse import quote, urlparse, urlunparse

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
    index_target_group = parser.add_argument_group("Index selection")
    index_target_group.add_argument(
        "-i",
        "--index",
        help="Redis index name to connect to",
        type=str,
        required=False,
    )
    index_target_group.add_argument(
        "-s",
        "--schema",
        help="Path to a schema YAML file",
        type=str,
        required=False,
    )

    redis_group = parser.add_argument_group("Redis connection options")
    redis_group.add_argument(
        "-u",
        "--url",
        help="Redis URL for data-plane commands",
        type=str,
        required=False,
    )
    redis_group.add_argument(
        "--host",
        help="Redis host for data-plane commands",
        type=str,
        default=None,
    )
    redis_group.add_argument(
        "-p",
        "--port",
        help="Redis port for data-plane commands",
        type=int,
        default=None,
    )
    redis_group.add_argument(
        "--user",
        help="Redis username for data-plane commands",
        type=str,
        default=None,
    )
    redis_group.add_argument("--ssl", help="Use SSL for Redis", action="store_true")
    redis_group.add_argument(
        "-a",
        "--password",
        help="Redis password for data-plane commands",
        type=str,
        default=None,
    )
    return parser


def _cli_json_default(obj: object) -> object:
    """Handle common non-JSON-native types (e.g. bytes from Redis) for cli_print_json."""
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def add_json_output_flag(parser: ArgumentParser) -> ArgumentParser:
    """Register ``--json`` for machine-readable stdout (one JSON object on success)."""
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json",
        default=False,
        help="Format output (when successful) as machine-readable JSON",
    )
    return parser


def cli_print_json(data: Mapping[str, Any]) -> None:
    """Write a single JSON object to stdout (deterministic key order for tests and scripts)."""
    print(json.dumps(data, default=_cli_json_default))
