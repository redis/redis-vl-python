import os
from argparse import Action, ArgumentParser, Namespace

from redisvl.redis.constants import REDIS_URL_ENV_VAR
from redisvl.utils.log import get_logger

logger = get_logger("[RedisVL]")
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
_TRACKED_CONNECTION_OPTIONS_ATTR = "_tracked_connection_options"
_TRACK_CONNECTION_OPTIONS_ATTR = "_track_connection_options"


def _get_tracked_connection_options(args: Namespace) -> set[str]:
    return set(getattr(args, _TRACKED_CONNECTION_OPTIONS_ATTR, ()))


def _is_tracking_connection_options(args: Namespace) -> bool:
    return bool(getattr(args, _TRACK_CONNECTION_OPTIONS_ATTR, False))


def _mark_connection_option(args: Namespace, option: str) -> None:
    tracked_options = _get_tracked_connection_options(args)
    tracked_options.add(option)
    setattr(args, _TRACKED_CONNECTION_OPTIONS_ATTR, tracked_options)


class _StoreTrackedOption(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        _mark_connection_option(namespace, self.dest)
        setattr(namespace, self.dest, values)


class _StoreTrackedTrue(Action):
    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super().__init__(
            option_strings,
            dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        _mark_connection_option(namespace, self.dest)
        setattr(namespace, self.dest, True)


def _has_explicit_connection_options(args: Namespace) -> bool:
    if _is_tracking_connection_options(args):
        return bool(
            _get_tracked_connection_options(args)
            & {"host", "port", "user", "password", "ssl"}
        )

    return any(
        getattr(args, attribute, None) is not None
        for attribute in ("host", "port", "user", "password")
    ) or bool(getattr(args, "ssl", False))


def _get_auth_credentials(args: Namespace) -> tuple[str | None, str | None]:
    if _is_tracking_connection_options(args):
        tracked_options = _get_tracked_connection_options(args)
        user = getattr(args, "user", None) if "user" in tracked_options else None
        password = (
            getattr(args, "password", None) if "password" in tracked_options else None
        )
        return user, password

    return getattr(args, "user", None), getattr(args, "password", None)


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
    parser.set_defaults(
        **{
            _TRACK_CONNECTION_OPTIONS_ATTR: True,
            _TRACKED_CONNECTION_OPTIONS_ATTR: (),
        }
    )
    parser.add_argument("-i", "--index", help="Index name", type=str, required=False)
    parser.add_argument(
        "-s", "--schema", help="Path to schema file", type=str, required=False
    )
    parser.add_argument("-u", "--url", help="Redis URL", type=str, required=False)
    parser.add_argument(
        "--host",
        help="Redis host",
        type=str,
        default=DEFAULT_REDIS_HOST,
        action=_StoreTrackedOption,
    )
    parser.add_argument(
        "-p",
        "--port",
        help="Redis port",
        type=int,
        default=DEFAULT_REDIS_PORT,
        action=_StoreTrackedOption,
    )
    parser.add_argument(
        "--user",
        help="Redis username",
        type=str,
        default="default",
        action=_StoreTrackedOption,
    )
    parser.add_argument("--ssl", help="Use SSL", action=_StoreTrackedTrue)
    parser.add_argument(
        "-a",
        "--password",
        help="Redis password",
        type=str,
        default="",
        action=_StoreTrackedOption,
    )
    return parser
