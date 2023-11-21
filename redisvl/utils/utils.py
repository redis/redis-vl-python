from functools import wraps
from typing import Any, List

import numpy as np


def make_dict(values: List[Any]):
    # TODO make this a real function
    i = 0
    di = {}
    while i < len(values) - 1:
        di[values[i]] = values[i + 1]
        i += 2
    return di


def convert_bytes(data: Any) -> Any:
    if isinstance(data, bytes):
        return data.decode("ascii")
    if isinstance(data, dict):
        return dict(map(convert_bytes, data.items()))
    if isinstance(data, list):
        return list(map(convert_bytes, data))
    if isinstance(data, tuple):
        return map(convert_bytes, data)
    return data


# required modules
REDIS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20400},
    {"name": "searchlight", "ver": 20400},
]


def check_redis_modules_exist(client) -> None:
    """Check if the correct Redis modules are installed."""
    installed_modules = client.module_list()
    installed_modules = {
        module[b"name"].decode("utf-8"): module for module in installed_modules
    }
    for module in REDIS_REQUIRED_MODULES:
        if module["name"] in installed_modules and int(
            installed_modules[module["name"]][b"ver"]
        ) >= int(
            module["ver"]
        ):  # type: ignore[call-overload]
            return
    # otherwise raise error
    error_message = (
        "You must add the RediSearch (>= 2.4) module from Redis Stack. "
        "Please refer to Redis Stack docs: https://redis.io/docs/stack/"
    )
    raise ValueError(error_message)


async def check_async_redis_modules_exist(client) -> None:
    """Check if the correct Redis modules are installed."""
    installed_modules = await client.module_list()
    installed_modules = {
        module[b"name"].decode("utf-8"): module for module in installed_modules
    }
    for module in REDIS_REQUIRED_MODULES:
        if module["name"] in installed_modules and int(
            installed_modules[module["name"]][b"ver"]
        ) >= int(
            module["ver"]
        ):  # type: ignore[call-overload]
            return
    # otherwise raise error
    error_message = (
        "You must add the RediSearch (>= 2.4) module from Redis Stack. "
        "Please refer to Redis Stack docs: https://redis.io/docs/stack/"
    )
    raise ValueError(error_message)


def check_modules_present(client_variable_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            client = getattr(self, client_variable_name)
            check_redis_modules_exist(client)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def check_async_modules_present(client_variable_name: str):
    async def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            client = getattr(self, client_variable_name)
            await check_redis_modules_exist(client)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def array_to_buffer(array: List[float], dtype: Any = np.float32) -> bytes:
    """Convert a list of floats into a numpy byte string."""
    return np.array(array).astype(dtype).tobytes()
