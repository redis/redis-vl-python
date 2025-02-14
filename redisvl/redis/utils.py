import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
from ml_dtypes import bfloat16

from redisvl.schema.fields import VectorDataType


def make_dict(values: List[Any]) -> Dict[Any, Any]:
    """Convert a list of objects into a dictionary"""
    i = 0
    di = {}
    while i < len(values) - 1:
        di[values[i]] = values[i + 1]
        i += 2
    return di


def convert_bytes(data: Any) -> Any:
    """Convert bytes data back to string"""
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except:
            return data
    if isinstance(data, dict):
        return {convert_bytes(key): convert_bytes(value) for key, value in data.items()}
    if isinstance(data, list):
        return [convert_bytes(item) for item in data]
    if isinstance(data, tuple):
        return tuple(convert_bytes(item) for item in data)
    return data


def array_to_buffer(array: List[float], dtype: str) -> bytes:
    """Convert a list of floats into a numpy byte string."""
    try:
        VectorDataType(dtype.upper())
    except ValueError:
        raise ValueError(
            f"Invalid data type: {dtype}. Supported types are: {[t.lower() for t in VectorDataType]}"
        )
    return np.array(array, dtype=dtype.lower()).tobytes()


def buffer_to_array(buffer: bytes, dtype: str) -> List[Any]:
    """Convert bytes into into a list of numerics."""
    try:
        VectorDataType(dtype.upper())
    except ValueError:
        raise ValueError(
            f"Invalid data type: {dtype}. Supported types are: {[t.lower() for t in VectorDataType]}"
        )
    return np.frombuffer(buffer, dtype=dtype.lower()).tolist()  # type: ignore[return-value]


def hashify(content: str, extras: Optional[Dict[str, Any]] = None) -> str:
    """Create a secure hash of some arbitrary input text and optional dictionary."""
    if extras:
        extra_string = " ".join([str(k) + str(v) for k, v in sorted(extras.items())])
        content = content + extra_string
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
