from functools import wraps
import json
from enum import Enum
from time import time
from typing import Any, Callable, Dict, Optional
from uuid import uuid4
from warnings import warn

from pydantic.v1 import BaseModel


def create_uuid() -> str:
    """Generate a unique indentifier to group related Redis documents."""
    return str(uuid4())


def current_timestamp() -> float:
    """Generate a unix epoch timestamp to assign to Redis documents."""
    return time()


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """
    Custom serialization function that converts a Pydantic model to a dict,
    serializing Enum fields to their values, and handling nested models and lists.
    """

    def serialize_item(item):
        if isinstance(item, Enum):
            return item.value.lower()
        elif isinstance(item, dict):
            return {key: serialize_item(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [serialize_item(element) for element in item]
        else:
            return item

    serialized_data = model.dict(exclude_none=True)
    for key, value in serialized_data.items():
        serialized_data[key] = serialize_item(value)
    return serialized_data


def validate_vector_dims(v1: int, v2: int) -> None:
    """Check the equality of vector dimensions."""
    if v1 != v2:
        raise ValueError(
            "Invalid vector dimensions! " f"Vector has dims defined as {v1}",
            f"Vector field has dims defined as {v2}",
            "Vector dims must be equal in order to perform similarity search.",
        )


def serialize(data: Dict[str, Any]) -> str:
    """Serlize the input into a string."""
    return json.dumps(data)


def deserialize(data: str) -> Dict[str, Any]:
    """Deserialize the input from a string."""
    return json.loads(data)


def deprecated_argument(argument: str, replacement: Optional[str] = None) -> Callable:
    """
    Decorator to warn if a deprecated argument is passed.

    When the wrapped function is called, the decorator will warn if the
    deprecated argument is passed as an argument or keyword argument.
    """

    message = f"Argument {argument} is deprecated and will be removed in the next major release."
    if replacement:
        message += f" Use {replacement} instead."

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            argument_names = func.__code__.co_varnames

            if argument in argument_names:
                warn(message, DeprecationWarning, stacklevel=2)
            elif argument in kwargs:
                warn(message, DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)

        return inner

    return wrapper
