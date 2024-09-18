import hashlib
from typing import Any, Dict, List

import numpy as np
from ml_dtypes import bfloat16

VectorDataTypes = {
    "BFLOAT16": bfloat16,
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}


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
        dtype = VectorDataTypes[dtype.upper()]
    except KeyError:
        raise ValueError(
            f"Invalid data type: {dtype}. Supported types are: {VectorDataTypes.keys()}"
        )
    return np.array(array).astype(dtype).tobytes()


def buffer_to_array(buffer: bytes, dtype: str) -> List[float]:
    """Convert bytes into into a list of floats."""
    try:
        dtype = VectorDataTypes[dtype.upper()]
    except KeyError:
        raise ValueError(
            f"Invalid data type: {dtype}. Supported types are: {VectorDataTypes.keys()}"
        )
    return np.frombuffer(buffer, dtype=dtype).tolist()


def hashify(content: str) -> str:
    """Create a secure hash of some arbitrary input text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
