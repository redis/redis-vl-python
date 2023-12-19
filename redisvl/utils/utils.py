from typing import Any, List

import numpy as np
from IPython.display import HTML, display
from redis.commands.search.result import Result


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


def array_to_buffer(array: List[float], dtype: Any = np.float32) -> bytes:
    """Convert a list of floats into a numpy byte string."""
    return np.array(array).astype(dtype).tobytes()


def table_print(dict_list):
    # If there's nothing in the list, there's nothing to print
    if len(dict_list) == 0:
        return

    # Getting column names (dictionary keys) using the first dictionary
    columns = dict_list[0].keys()

    # HTML table header
    html = "<table><tr><th>"
    html += "</th><th>".join(columns)
    html += "</th></tr>"

    # HTML table content
    for dictionary in dict_list:
        html += "<tr><td>"
        html += "</td><td>".join(str(dictionary[column]) for column in columns)
        html += "</td></tr>"

    # HTML table footer
    html += "</table>"

    # Displaying the table
    display(HTML(html))


def result_print(results):
    if isinstance(results, Result):
        # If there's nothing in the list, there's nothing to print
        if len(results.docs) == 0:
            return

        results = [doc.__dict__ for doc in results.docs]

    to_remove = ["id", "payload"]
    for doc in results:
        for key in to_remove:
            if key in doc:
                del doc[key]

    table_print(results)
