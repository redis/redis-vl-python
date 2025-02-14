import inspect
import json
import warnings
from contextlib import ContextDecorator, contextmanager
from enum import Enum
from functools import wraps
from time import time
from typing import Any, Callable, Dict, Optional
from warnings import warn

from pydantic.v1 import BaseModel
from ulid import ULID


def create_ulid() -> str:
    """Generate a unique indentifier to group related Redis documents."""
    return str(ULID())


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

    NOTE: The @deprecated_argument decorator should not fall "outside"
    of the @classmethod decorator, but instead should come between
    it and the method definition. For example:

        class MyClass:
            @classmethod
            @deprecated_argument("old_arg", "new_arg")
            @other_decorator
            def test_method(cls, old_arg=None, new_arg=None):
                pass
    """
    message = f"Argument {argument} is deprecated and will be removed in the next major release."
    if replacement:
        message += f" Use {replacement} instead."

    def decorator(func):
        # Check if the function is a classmethod or staticmethod
        if isinstance(func, (classmethod, staticmethod)):
            underlying = func.__func__

            @wraps(underlying)
            def inner_wrapped(*args, **kwargs):
                if argument in kwargs:
                    warn(message, DeprecationWarning, stacklevel=2)
                else:
                    sig = inspect.signature(underlying)
                    bound_args = sig.bind(*args, **kwargs)
                    if argument in bound_args.arguments:
                        warn(message, DeprecationWarning, stacklevel=2)
                return underlying(*args, **kwargs)

            if isinstance(func, classmethod):
                return classmethod(inner_wrapped)
            else:
                return staticmethod(inner_wrapped)
        else:

            @wraps(func)
            def inner_normal(*args, **kwargs):
                if argument in kwargs:
                    warn(message, DeprecationWarning, stacklevel=2)
                else:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    if argument in bound_args.arguments:
                        warn(message, DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)

            return inner_normal

    return decorator


@contextmanager
def assert_no_warnings():
    """
    Assert that a function does not emit any warnings when called.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
