import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Pattern

if TYPE_CHECKING:
    from redis.commands.search.result import Result
    from redis.commands.search.document import Document

import numpy as np


class TokenEscaper:
    """
    Escape punctuation within an input string. Taken from RedisOM Python.
    """

    # Characters that RediSearch requires us to escape during queries.
    # Source: https://redis.io/docs/stack/search/reference/escaping/#the-rules-of-text-field-tokenization
    DEFAULT_ESCAPED_CHARS = r"[,.<>{}\[\]\\\"\':;!@#$%^&*()\-+=~\/ ]"

    def __init__(self, escape_chars_re: Optional[Pattern] = None):
        if escape_chars_re:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)

    def escape(self, value: str) -> str:
        def escape_symbol(match):
            value = match.group(0)
            return f"\\{value}"

        return self.escaped_chars_re.sub(escape_symbol, value)


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


def array_to_buffer(array: List[float], dtype: Any = np.float32) -> bytes:
    """Convert a list of floats into a numpy byte string."""
    return np.array(array).astype(dtype).tobytes()


def process_results(results: List["Result"]) -> List[Dict[str, Any]]:
    """Convert a list of search Result objects into a list of document dicts"""

    def _process(doc: "Document") -> dict:
        d = doc.__dict__
        if "payload" in d:
            del d["payload"]
        return d

    return [_process(doc) for doc in results.docs]
