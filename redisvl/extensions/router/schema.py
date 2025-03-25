import warnings
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Annotated

from redisvl.extensions.constants import ROUTE_VECTOR_FIELD_NAME
from redisvl.schema import IndexSchema


class Route(BaseModel):
    """Model representing a routing path with associated metadata and thresholds."""

    name: str
    """The name of the route."""
    references: List[str]
    """List of reference phrases for the route."""
    metadata: Dict[str, Any] = Field(default={})
    """Metadata associated with the route."""
    distance_threshold: Annotated[float, Field(strict=True, gt=0, le=2)] = 0.5
    """Distance threshold for matching the route."""

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Route name must not be empty")
        return v

    @field_validator("references")
    @classmethod
    def references_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("References must not be empty")
        if any(not ref.strip() for ref in v):
            raise ValueError("All references must be non-empty strings")
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

    """The maximum number of top matches to return."""
    max_k: Annotated[int, Field(strict=True, default=1, gt=0)] = 1
    """Aggregation method to use to classify queries."""
    aggregation_method: DistanceAggregationMethod = Field(
        default=DistanceAggregationMethod.avg
    )

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def remove_distance_threshold(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "distance_threshold" in values:
            warnings.warn(
                "The 'distance_threshold' field is deprecated and will be ignored. Set distance_threshold per Route.",
                DeprecationWarning,
                stacklevel=2,
            )
            values.pop("distance_threshold")
        return values


class SemanticRouterIndexSchema(IndexSchema):
    """Customized index schema for SemanticRouter."""

    @classmethod
    def from_params(cls, name: str, vector_dims: int, dtype: str):
        """Create an index schema based on router name and vector dimensions.

        Args:
            name (str): The name of the index.
            vector_dims (int): The dimensions of the vectors.

        Returns:
            SemanticRouterIndexSchema: The constructed index schema.
        """
        return cls(
            index={"name": name, "prefix": name},  # type: ignore
            fields=[  # type: ignore
                {"name": "route_name", "type": "tag"},
                {"name": "reference", "type": "text"},
                {
                    "name": ROUTE_VECTOR_FIELD_NAME,
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": vector_dims,
                        "distance_metric": "cosine",
                        "datatype": dtype,
                    },
                },
            ],
        )
