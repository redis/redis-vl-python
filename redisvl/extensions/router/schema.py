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
    model: Optional[str] = None
    """Optional LiteLLM-compatible model identifier for LLM routing (e.g., 'openai/gpt-4.1-nano')."""

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
    model: Optional[str] = None
    """The LiteLLM model identifier (populated when route has a model field)."""
    confidence: Optional[float] = None
    """Routing confidence score (1 - distance/2), range 0-1."""
    alternatives: List[tuple] = Field(default_factory=list)
    """Alternative route matches as (route_name, distance) tuples."""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Route metadata (costs, capabilities, etc.)."""

    @property
    def tier(self) -> Optional[str]:
        """Alias for name (backward compatibility with LLMRouteMatch)."""
        return self.name

    def __bool__(self) -> bool:
        """Return True if a route was matched."""
        return self.name is not None


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
    max_k: Annotated[int, Field(strict=True, gt=0)] = 1
    """Aggregation method to use to classify queries."""
    aggregation_method: DistanceAggregationMethod = Field(
        default=DistanceAggregationMethod.avg
    )
    cost_optimization: bool = False
    """Whether to prefer cheaper routes/models when distances are close."""
    cost_weight: Annotated[float, Field(ge=0, le=1)] = 0.1
    """Weight for cost penalty in routing (0 = ignore cost, 1 = cost dominates)."""
    default_route: Optional[str] = None
    """Route name to use when no match found (None = return no match)."""

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
                {"name": "reference_id", "type": "tag"},
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


class PretrainedReference(BaseModel):
    """A reference with pre-computed embedding vector.

    Used for exporting/importing routers with embeddings to avoid
    re-computing embeddings on load.
    """

    text: str
    """The reference text."""

    vector: List[float]
    """Pre-computed embedding vector."""


class PretrainedRoute(BaseModel):
    """A route with pre-computed embeddings for all references.

    Used in pretrained router configurations.
    """

    model_config = ConfigDict(protected_namespaces=())

    name: str
    """Route name."""

    references: List[PretrainedReference]
    """References with pre-computed vectors."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Route metadata."""

    distance_threshold: Annotated[float, Field(strict=True, gt=0, le=2)] = 0.5
    """Distance threshold."""

    model: Optional[str] = None
    """Optional LiteLLM model identifier."""


class PretrainedRouterConfig(BaseModel):
    """Complete router configuration with pre-computed embeddings.

    This format is used for distributing pretrained routers that can be
    loaded without needing to re-embed all references.
    """

    name: str
    """Router name."""

    version: str = "1.0.0"
    """Configuration version."""

    vectorizer: Dict[str, Any]
    """Vectorizer configuration (type, model name)."""

    routes: List[PretrainedRoute]
    """Routes with pre-computed embeddings."""

    routing_config: Dict[str, Any] = Field(default_factory=dict)
    """Routing configuration."""

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_tiers(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Handle legacy 'tiers' field for backward compatibility."""
        if "tiers" in values and "routes" not in values:
            values["routes"] = values.pop("tiers")
        return values
