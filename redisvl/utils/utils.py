import asyncio
import inspect
import json
import logging
import warnings
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from time import time
from typing import Any, Callable, Coroutine, Dict, Optional
from warnings import warn

from pydantic import BaseModel
from ulid import ULID


def create_ulid() -> str:
    """Generate a unique identifier to group related Redis documents."""
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

    serialized_data = model.model_dump(exclude_none=True)
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


def deprecated_function(name: Optional[str] = None, replacement: Optional[str] = None):
    """
    Decorator to mark a function as deprecated.

    When the wrapped function is called, the decorator will log a deprecation
    warning.
    """

    def decorator(func):
        fn_name = name or func.__name__
        warning_message = (
            f"Function {fn_name} is deprecated and will be "
            "removed in the next major release. "
        )
        if replacement:
            warning_message += replacement

        @wraps(func)
        def wrapper(*args, **kwargs):
            warn(warning_message, category=DeprecationWarning, stacklevel=3)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def sync_wrapper(fn: Callable[[], Coroutine[Any, Any, Any]]) -> Callable[[], None]:
    def wrapper():
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        try:
            if loop is None or not loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            task = loop.create_task(fn())
            loop.run_until_complete(task)
        except RuntimeError:
            # This could happen if an object stored an event loop and now
            # that event loop is closed. There's nothing we can do other than
            # advise the user to use explicit cleanup methods.
            #
            # Uses logging module instead of get_logger() to avoid I/O errors
            # if the wrapped function is called as a finalizer.
            logging.info(
                f"Could not run the async function {fn.__name__} because the event loop is closed. "
                "This usually means the object was not properly cleaned up. Please use explicit "
                "cleanup methods (e.g., disconnect(), close()) or use the object as an async "
                "context manager.",
            )
            return

    return wrapper


def norm_cosine_distance(value: float) -> float:
    """
    Normalize a cosine distance to a similarity score between 0 and 1.
    """
    return max((2 - value) / 2, 0)


def denorm_cosine_distance(value: float) -> float:
    """
    Denormalize a similarity score between 0 and 1 to a cosine distance between
    0 and 2.
    """
    return max(2 - 2 * value, 0)


def norm_l2_distance(value: float) -> float:
    """
    Normalize the L2 distance.
    """
    return 1 / (1 + value)
