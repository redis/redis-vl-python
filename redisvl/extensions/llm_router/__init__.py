"""LLM Router extension for intelligent model selection.

This module provides semantic routing for LLM model tier selection.
Routes queries to the most appropriate model based on semantic similarity
to reference phrases.

Example:
    >>> from redisvl.extensions.llm_router import LLMRouter, ModelTier
    >>> 
    >>> tiers = [
    ...     ModelTier(
    ...         name="simple",
    ...         model="anthropic/claude-haiku-4-5",
    ...         references=["hello", "hi", "thanks"],
    ...         distance_threshold=0.5,
    ...     ),
    ...     ModelTier(
    ...         name="reasoning", 
    ...         model="anthropic/claude-sonnet-4-5",
    ...         references=["analyze this", "explain how"],
    ...         distance_threshold=0.6,
    ...     ),
    ... ]
    >>> 
    >>> router = LLMRouter(
    ...     name="my-router",
    ...     tiers=tiers,
    ...     redis_url="redis://localhost:6379",
    ... )
    >>> 
    >>> match = router.route("hello, how are you?")
    >>> print(f"Use {match.model} for this query")
"""

from redisvl.extensions.llm_router.router import LLMRouter
from redisvl.extensions.llm_router.schema import (
    DistanceAggregationMethod,
    LLMRouteMatch,
    ModelTier,
    PretrainedReference,
    PretrainedRouterConfig,
    PretrainedTier,
    RoutingConfig,
)

__all__ = [
    "LLMRouter",
    "ModelTier",
    "LLMRouteMatch",
    "RoutingConfig",
    "DistanceAggregationMethod",
    "PretrainedReference",
    "PretrainedTier",
    "PretrainedRouterConfig",
]
