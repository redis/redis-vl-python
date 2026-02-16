"""Schema definitions for LLM Router.

This module defines the Pydantic models for model tiers, routing configuration,
and route match results.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Annotated


class ModelTier(BaseModel):
    """Model representing a routing tier with associated LLM model and metadata.
    
    A ModelTier defines a category of queries that should be routed to a specific
    LLM model. Each tier has reference phrases that define its "semantic surface area"
    and metadata about the model's capabilities and costs.
    
    Attributes:
        name: Unique identifier for the tier (e.g., "simple", "reasoning", "expert")
        model: LiteLLM-compatible model identifier (e.g., "anthropic/claude-haiku-4-5")
        references: Example phrases that should route to this tier
        metadata: Model configuration including costs and capabilities
        distance_threshold: Maximum cosine distance for matching (0-2, lower is stricter)
    
    Example:
        >>> tier = ModelTier(
        ...     name="simple",
        ...     model="anthropic/claude-haiku-4-5",
        ...     references=["hello", "hi there", "thanks"],
        ...     metadata={"cost_per_1k_input": 0.00025},
        ...     distance_threshold=0.5,
        ... )
    """

    model_config = ConfigDict(protected_namespaces=())

    name: str
    """Unique identifier for the tier."""
    
    model: str
    """LiteLLM-compatible model identifier (e.g., 'anthropic/claude-sonnet-4-5')."""
    
    references: List[str]
    """Example phrases that define this tier's semantic space."""
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Model metadata including costs, capabilities, provider info."""
    
    distance_threshold: Annotated[float, Field(strict=True, gt=0, le=2)] = 0.5
    """Maximum cosine distance for matching this tier (Redis COSINE: 0-2)."""

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Tier name must not be empty")
        return v

    @field_validator("references")
    @classmethod
    def references_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("References must not be empty")
        if any(not ref.strip() for ref in v):
            raise ValueError("All references must be non-empty strings")
        return v

    @field_validator("model")
    @classmethod
    def model_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Model must not be empty")
        return v


class LLMRouteMatch(BaseModel):
    """Result of routing a query to an LLM model tier.
    
    Contains information about which tier was selected, the model to use,
    and routing metadata like distance and confidence scores.
    
    Attributes:
        tier: Name of the matched tier (None if no match)
        model: LiteLLM model identifier to use
        distance: Cosine distance to the matched tier (lower is better)
        confidence: Routing confidence score (0-1, higher is better)
        alternatives: Other possible tier matches with their distances
    """

    tier: Optional[str] = None
    """Name of the matched tier."""
    
    model: Optional[str] = None
    """LiteLLM model identifier."""
    
    distance: Optional[float] = None
    """Cosine distance to the matched tier references."""
    
    confidence: Optional[float] = None
    """Routing confidence (1 - distance/2), range 0-1."""
    
    alternatives: List[tuple] = Field(default_factory=list)
    """Alternative tier matches as (tier_name, distance) tuples."""
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Tier metadata (costs, capabilities, etc.)."""

    def __bool__(self) -> bool:
        """Return True if a tier was matched."""
        return self.tier is not None


class DistanceAggregationMethod(Enum):
    """Method for aggregating distances across multiple references."""

    avg = "avg"
    """Average of distances to all matching references."""
    
    min = "min"
    """Minimum distance (closest reference)."""
    
    sum = "sum"
    """Sum of distances."""


class RoutingConfig(BaseModel):
    """Configuration for LLM routing behavior.
    
    Attributes:
        max_k: Maximum number of tier matches to return
        aggregation_method: How to aggregate distances across references
        cost_optimization: Whether to prefer cheaper tiers when distances are close
        cost_weight: Weight for cost in routing decisions (0-1)
        default_tier: Tier to use when no match is found
    """

    model_config = ConfigDict(extra="ignore")

    max_k: Annotated[int, Field(strict=True, gt=0)] = 1
    """Maximum number of tier matches to return."""
    
    aggregation_method: DistanceAggregationMethod = Field(
        default=DistanceAggregationMethod.avg
    )
    """Method for aggregating distances."""
    
    cost_optimization: bool = False
    """Whether to prefer cheaper tiers when distances are close."""
    
    cost_weight: Annotated[float, Field(ge=0, le=1)] = 0.1
    """Weight for cost penalty in routing (0 = ignore cost, 1 = cost dominates)."""
    
    default_tier: Optional[str] = None
    """Tier to use when no match found (None = return no match)."""


class PretrainedReference(BaseModel):
    """A reference with pre-computed embedding vector.
    
    Used for exporting/importing routers with embeddings to avoid
    re-computing embeddings on load.
    """
    
    text: str
    """The reference text."""
    
    vector: List[float]
    """Pre-computed embedding vector."""


class PretrainedTier(BaseModel):
    """A tier with pre-computed embeddings for all references.
    
    Used in pretrained router configurations.
    """
    
    model_config = ConfigDict(protected_namespaces=())
    
    name: str
    """Tier name."""
    
    model: str
    """LiteLLM model identifier."""
    
    references: List[PretrainedReference]
    """References with pre-computed vectors."""
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Tier metadata."""
    
    distance_threshold: float = 0.5
    """Distance threshold."""


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
    
    tiers: List[PretrainedTier]
    """Tiers with pre-computed embeddings."""
    
    routing_config: Dict[str, Any] = Field(default_factory=dict)
    """Routing configuration."""
