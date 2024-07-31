from enum import Enum
from time import time
from typing import Any, Dict
from uuid import uuid4

from pydantic.v1 import BaseModel


def create_uuid() -> str:
    return str(uuid4())


def current_timestamp():
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
