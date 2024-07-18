from enum import Enum
from typing import Dict, List, Optional

from pydantic.v1 import BaseModel, Field, validator


class Route(BaseModel):
    """Model representing a routing path with associated metadata and thresholds."""

    name: str
    """The name of the route."""
    references: List[str]
    """List of reference phrases for the route."""
    metadata: Dict[str, str] = Field(default={})
    """Metadata associated with the route."""
    distance_threshold: Optional[float] = Field(default=None)
    """Distance threshold for matching the route."""

    @validator("name")
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Route name must not be empty")
        return v

    @validator("references")
    def references_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("References must not be empty")
        if any(not ref.strip() for ref in v):
            raise ValueError("All references must be non-empty strings")
        return v

    @validator("distance_threshold")
    def distance_threshold_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Route distance threshold must be greater than zero")
        return v


class RouteMatch(BaseModel):
    """Model representing a matched route with distance information."""

    name: Optional[str] = None
    """The matched route name."""
    distance: Optional[float] = Field(default=None)
    """The vector distance between the statement and the matched route."""


class DistanceAggregationMethod(Enum):
    """Enumeration for distance aggregation methods."""

    avg = "avg"
    """Compute the average of the vector distances."""
    min = "min"
    """Compute the minimum of the vector distances."""
    sum = "sum"
    """Compute the sum of the vector distances."""


class RoutingConfig(BaseModel):
    """Configuration for routing behavior."""

    distance_threshold: float = Field(default=0.5)
    """The threshold for semantic distance."""
    max_k: int = Field(default=1)
    """The maximum number of top matches to return."""
    aggregation_method: DistanceAggregationMethod = Field(
        default=DistanceAggregationMethod.avg
    )
    """Aggregation method to use to classify queries."""

    @validator("max_k")
    def max_k_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("max_k must be a positive integer")
        return v

    @validator("distance_threshold")
    def distance_threshold_must_be_valid(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("distance_threshold must be between 0 and 1")
        return v
