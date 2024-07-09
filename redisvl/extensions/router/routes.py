

from pydantic.v1 import BaseModel, Field, validator
from typing import List, Dict, Optional

from enum import Enum


class Route(BaseModel):
    name: str
    """The name of the route"""
    references: List[str]
    """List of reference phrases for the route"""
    metadata: Dict[str, str] = Field(default={})
    """Metadata associated with the route"""

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Route name must not be empty')
        return v

    @validator('references')
    def references_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('References must not be empty')
        if any(not ref.strip() for ref in v):
            raise ValueError('All references must be non-empty strings')
        return v


class AccumulationMethod(Enum):
    # TODO: tidy up the enum usage
    simple = "simple" # Take the winner at face value
    avg = "avg" # Consider the avg score of all matches
    sum = "sum" # Consider the cumulative score of all matches
    auto = "auto" # Pick on the user's behalf?


class RoutingConfig(BaseModel):
    max_k: int = Field(default=1)
    """The maximum number of top matches to return"""
    distance_threshold: float = Field(default=0.5)
    """The threshold for semantic distance"""
    accumulation_method: AccumulationMethod = Field(default=AccumulationMethod.auto)
    """The accumulation method used to determine the matching route"""

    @validator('max_k')
    def max_k_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('max_k must be a positive integer')
        return v

    @validator('distance_threshold')
    def distance_threshold_must_be_valid(cls, v):
        if v is not None and (v <= 0 or v > 1):
            raise ValueError('distance_threshold must be between 0 and 1')
        return v

