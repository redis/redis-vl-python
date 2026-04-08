import os
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence

from redisvl.redis.constants import REDIS_URL_ENV_VAR
from redisvl.utils.log import get_logger

logger = get_logger("[RedisVL]")
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
_HOST_FLAGS = ("--host",)
_PORT_FLAGS = ("-p", "--port")
_USER_FLAGS = ("--user",)
_PASSWORD_FLAGS = ("-a", "--password")
_SSL_FLAGS = ("--ssl",)
_ALL_CONNECTION_FLAGS = (
    _HOST_FLAGS,
    _PORT_FLAGS,
    _USER_FLAGS,
    _PASSWORD_FLAGS,
    _SSL_FLAGS,
)


def _get_cli_argv(args: Namespace) -> tuple[str, ...]:
    raw_argv = getattr(args, "_argv", ())
    return tuple(raw_argv) if isinstance(raw_argv, Sequence) else ()


def _argv_has_flag(argv: Sequence[str], *flags: str) -> bool:
    return any(flag in argv for flag in flags)


def _has_explicit_connection_options(args: Namespace) -> bool:
    argv = _get_cli_argv(args)
    if argv:
        return any(_argv_has_flag(argv, *flags) for flags in _ALL_CONNECTION_FLAGS)

    return any(
        (
            getattr(args, "host", None) not in (None, DEFAULT_REDIS_HOST),
            getattr(args, "port", None) not in (None, DEFAULT_REDIS_PORT),
            getattr(args, "user", None) not in (None, "", "default"),
            bool(getattr(args, "password", None)),
            bool(getattr(args, "ssl", False)),
        )
    )


def _get_auth_credentials(args: Namespace) -> tuple[str | None, str | None]:
    argv = _get_cli_argv(args)
    if argv:
        user = args.user if _argv_has_flag(argv, *_USER_FLAGS) else None
        password = args.password if _argv_has_flag(argv, *_PASSWORD_FLAGS) else None
        return user, password

    user = getattr(args, "user", None)
    if user in (None, "", "default"):
        user = None

    password = getattr(args, "password", None) or None
    return user, password


def _build_redis_url(args: Namespace) -> str:
    scheme = "rediss" if getattr(args, "ssl", False) else "redis"
    host = getattr(args, "host", None) or DEFAULT_REDIS_HOST
    port = getattr(args, "port", None) or DEFAULT_REDIS_PORT
    user, password = _get_auth_credentials(args)

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
    parser.add_argument(
        "--host", help="Redis host", type=str, default=DEFAULT_REDIS_HOST
    )
    parser.add_argument(
        "-p", "--port", help="Redis port", type=int, default=DEFAULT_REDIS_PORT
    )
    parser.add_argument("--user", help="Redis username", type=str, default="default")
    parser.add_argument("--ssl", help="Use SSL", action="store_true")
    parser.add_argument(
        "-a",
        "--password",
        help="Redis password",
        type=str,
        default="",
    )
    return parser
