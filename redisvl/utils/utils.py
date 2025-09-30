import asyncio
import importlib
import inspect
import json
import logging
import sys
import warnings
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from time import time
from typing import Any, Callable, Coroutine, Dict, Optional, Sequence, TypeVar, cast
from warnings import warn

from pydantic import BaseModel
from redis import Redis
from ulid import ULID

T = TypeVar("T")


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
        elif isinstance(item, BaseModel):
            # Recursively serialize nested BaseModel instances with exclude_defaults=False
            nested_data = item.model_dump(exclude_none=True, exclude_defaults=False)
            return {key: serialize_item(value) for key, value in nested_data.items()}
        elif isinstance(item, dict):
            return {key: serialize_item(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [serialize_item(element) for element in item]
        else:
            return item

    # Use exclude_defaults=False to preserve all field attributes including new ones
    serialized_data = model.model_dump(exclude_none=True, exclude_defaults=False)
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


def serialize(data: Any) -> str:
    """Serlize the input into a string."""
    return json.dumps(data)


def deserialize(data: str) -> Any:
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
        # Check if the interpreter is shutting down
        if sys is None or getattr(sys, "_getframe", None) is None:
            # Interpreter is shutting down, skip cleanup
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        except Exception:
            # Any other exception during loop detection means we should skip cleanup
            return

        try:
            if loop is None or not loop.is_running():
                # Check if asyncio module is still available
                if asyncio is None:
                    return

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            task = loop.create_task(fn())
            loop.run_until_complete(task)
        except (RuntimeError, AttributeError, TypeError) as e:
            # This could happen if an object stored an event loop and now
            # that event loop is closed, or if asyncio modules are being
            # torn down during interpreter shutdown.
            #
            # Silently ignore - attempting to log during shutdown can cause
            # errors when logging handlers are being torn down.
            return
        except Exception:
            # Any other unexpected exception should be silently ignored during shutdown
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


def scan_by_pattern(
    redis_client: Redis,
    pattern: str,
) -> Sequence[str]:
    """
    Scan the Redis database for keys matching a specific pattern.

    Args:
        redis (Redis): The Redis client instance.
        pattern (str): The pattern to match keys against.

    Returns:
        List[str]: A dictionary containing the keys and their values.
    """
    from redisvl.redis.utils import convert_bytes

    return convert_bytes(list(redis_client.scan_iter(match=pattern)))


def lazy_import(module_path: str) -> Any:
    """
    Lazily import a module or object from a module only when it's actually used.

    This function helps reduce startup time and avoid unnecessary dependencies
    by only importing modules when they are actually needed.

    Args:
        module_path (str): The import path, e.g., "numpy" or "numpy.array"

    Returns:
        Any: The imported module or object, or a proxy that will import it when used

    Examples:
        >>> np = lazy_import("numpy")
        >>> # numpy is not imported yet
        >>> array = np.array([1, 2, 3])  # numpy is imported here

        >>> array_func = lazy_import("numpy.array")
        >>> # numpy is not imported yet
        >>> arr = array_func([1, 2, 3])  # numpy is imported here
    """
    parts = module_path.split(".")
    top_module_name = parts[0]

    # Check if the module is already imported and we're not trying to access a specific attribute
    if top_module_name in sys.modules and len(parts) == 1:
        return sys.modules[top_module_name]

    # Create a proxy class that will import the module when any attribute is accessed
    class LazyModule:
        def __init__(self, module_path: str):
            self._module_path = module_path
            self._module = None
            self._parts = module_path.split(".")

        def _import_module(self):
            """Import the module or attribute on first use"""
            if self._module is not None:
                return self._module

            try:
                # Import the base module
                base_module_name = self._parts[0]
                module = importlib.import_module(base_module_name)

                # If we're importing just the module, return it
                if len(self._parts) == 1:
                    self._module = module
                    return module

                # Otherwise, try to get the specified attribute or submodule
                obj = module
                for part in self._parts[1:]:
                    try:
                        obj = getattr(obj, part)
                    except AttributeError:
                        # Attribute doesn't exist - we'll raise this error when the attribute is accessed
                        return None

                self._module = obj
                return obj
            except ImportError as e:
                # Store the error to raise it when the module is accessed
                self._import_error = e
                return None

        def __getattr__(self, name: str) -> Any:
            # Import the module if it hasn't been imported yet
            if self._module is None:
                module = self._import_module()

                # If import failed, raise the appropriate error
                if module is None:
                    # Use direct dictionary access to avoid recursion
                    if "_import_error" in self.__dict__:
                        raise ImportError(
                            f"Failed to lazily import {self._module_path}: {self._import_error}"
                        )
                    else:
                        # This means we couldn't find the attribute in the module path
                        raise AttributeError(
                            f"module '{self._parts[0]}' has no attribute '{self._parts[1]}'"
                        )

            # If we have a module, get the requested attribute
            if hasattr(self._module, name):
                return getattr(self._module, name)

            # If the attribute doesn't exist, raise AttributeError
            raise AttributeError(
                f"module '{self._module_path}' has no attribute '{name}'"
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            # Import the module if it hasn't been imported yet
            if self._module is None:
                module = self._import_module()

                # If import failed, raise the appropriate error
                if module is None:
                    # Use direct dictionary access to avoid recursion
                    if "_import_error" in self.__dict__:
                        raise ImportError(
                            f"Failed to lazily import {self._module_path}: {self._import_error}"
                        )
                    else:
                        # This means we couldn't find the attribute in the module path
                        raise ImportError(
                            f"Failed to find {self._module_path}: module '{self._parts[0]}' has no attribute '{self._parts[1]}'"
                        )

            # If the imported object is callable, call it
            if callable(self._module):
                return self._module(*args, **kwargs)

            # If it's not callable, this is an error
            raise TypeError(f"{self._module_path} is not callable")

    return LazyModule(module_path)
