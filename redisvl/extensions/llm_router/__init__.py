"""LLM Router extension for intelligent model selection (DEPRECATED).

.. deprecated::
    The ``llm_router`` module is deprecated. Use :class:`redisvl.extensions.router.SemanticRouter`
    instead, which now includes all LLM routing functionality.

This module provides backward compatibility aliases for code using the old
LLMRouter API. All functionality has been consolidated into SemanticRouter.

Migration guide:

Old code::

    from redisvl.extensions.llm_router import LLMRouter, ModelTier

    tiers = [ModelTier(name="simple", model="openai/gpt-4.1-nano", references=[...])]
    router = LLMRouter(name="my-router", tiers=tiers)
    match = router.route("hello")

New code::

    from redisvl.extensions.router import SemanticRouter, Route

    routes = [Route(name="simple", model="openai/gpt-4.1-nano", references=[...])]
    router = SemanticRouter(name="my-router", routes=routes)
    match = router("hello")  # Callable pattern

Key changes:
- ``LLMRouter`` → ``SemanticRouter``
- ``AsyncLLMRouter`` → ``AsyncSemanticRouter``
- ``ModelTier`` → ``Route``
- ``LLMRouteMatch`` → ``RouteMatch``
- ``router.route(query)`` → ``router(query)`` (callable pattern)
- ``tiers=`` → ``routes=``
"""

import warnings
from typing import Any, Dict, List, Optional

# Issue deprecation warning when module is imported
warnings.warn(
    "redisvl.extensions.llm_router is deprecated and will be removed in a future version. "
    "Use redisvl.extensions.router.SemanticRouter instead. "
    "See migration guide: https://docs.redisvl.com/user_guide/llm_router.html",
    DeprecationWarning,
    stacklevel=2,
)

# Import new classes
from redisvl.extensions.router import AsyncSemanticRouter as _AsyncSemanticRouter
from redisvl.extensions.router import Route, RouteMatch
from redisvl.extensions.router import SemanticRouter as _SemanticRouter
from redisvl.extensions.router.schema import (
    DistanceAggregationMethod,
    PretrainedReference,
    PretrainedRoute,
    PretrainedRouterConfig,
)
from redisvl.types import AsyncRedisClient, SyncRedisClient
from redisvl.utils.vectorize.base import BaseVectorizer


# Backward compatibility wrapper that maps tiers → routes
class LLMRouter(_SemanticRouter):
    """Backward compatibility wrapper for LLMRouter.

    This class wraps SemanticRouter and maps old parameter names (tiers)
    to new ones (routes) for backward compatibility.
    """

    def __init__(
        self,
        name: str,
        tiers: Optional[List[Route]] = None,
        routes: Optional[List[Route]] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        routing_config: Optional[Any] = None,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        cost_optimization: bool = False,
        connection_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize LLMRouter (deprecated, use SemanticRouter).

        Args:
            name: Router name.
            tiers: (Deprecated) Use routes instead.
            routes: List of Route objects.
            vectorizer: Vectorizer for embeddings.
            routing_config: Configuration for routing behavior.
            redis_client: Redis client.
            redis_url: Redis URL.
            overwrite: Whether to overwrite existing index.
            cost_optimization: Enable cost-aware routing.
            connection_kwargs: Additional Redis connection arguments.
        """
        # Map tiers → routes for backward compatibility
        if tiers is not None and routes is None:
            routes = tiers
        elif routes is None:
            routes = []

        # Map cost_optimization to routing_config
        if cost_optimization and routing_config:
            routing_config.cost_optimization = True
        elif cost_optimization:
            from redisvl.extensions.router.schema import RoutingConfig

            routing_config = RoutingConfig(cost_optimization=True)

        # Handle mutable default
        if connection_kwargs is None:
            connection_kwargs = {}

        # Call parent __init__
        super().__init__(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
            redis_client=redis_client,
            redis_url=redis_url,
            overwrite=overwrite,
            connection_kwargs=connection_kwargs,
            **kwargs,
        )

    def route(self, query: Optional[str] = None, **kwargs):
        """Route a query (deprecated method, use __call__ instead)."""
        return self(statement=query, **kwargs)

    @property
    def tiers(self):
        """Alias for routes (backward compatibility)."""
        return self.routes

    @tiers.setter
    def tiers(self, value):
        """Alias for routes (backward compatibility)."""
        self.routes = value

    @property
    def tier_names(self):
        """Alias for route_names (backward compatibility)."""
        return self.route_names

    @property
    def tier_thresholds(self):
        """Alias for route_thresholds (backward compatibility)."""
        return self.route_thresholds

    @property
    def default_tier(self):
        """Alias for default_route (backward compatibility)."""
        return self.routing_config.default_route

    def get_tier(self, tier_name: str):
        """Alias for get (backward compatibility)."""
        return self.get(tier_name)

    def add_tier(self, tier: Route):
        """Add a new tier (backward compatibility)."""
        if self.get(tier.name):
            raise ValueError(f"Tier {tier.name} already exists")
        self._add_routes([tier])
        self._update_router_state()

    def remove_tier(self, tier_name: str):
        """Remove a tier (backward compatibility)."""
        return self.remove_route(tier_name)

    def add_tier_references(self, tier_name: str, references):
        """Add references to a tier (backward compatibility)."""
        return self.add_route_references(tier_name, references)

    def update_tier_threshold(self, tier_name: str, threshold: float):
        """Update a tier's distance threshold (backward compatibility)."""
        route = self.get(tier_name)
        if route is None:
            raise ValueError(f"Tier {tier_name} not found")
        if not (0 < threshold <= 2):
            raise ValueError("Threshold must be in range (0, 2]")
        route.distance_threshold = threshold  # type: ignore
        self._update_router_state()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with 'tiers' for backward compatibility."""
        result = super().to_dict()
        # Map routes → tiers for backward compatibility
        result["tiers"] = result.pop("routes")
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs):
        """Load from dict with 'tiers' for backward compatibility."""
        # Map tiers → routes for backward compatibility
        if "tiers" in data and "routes" not in data:
            data = data.copy()
            data["routes"] = data.pop("tiers")
        return super().from_dict(data, **kwargs)

    @classmethod
    def from_yaml(cls, file_path: str, **kwargs):
        """Load from YAML with 'tiers' for backward compatibility."""
        # The parent from_yaml calls from_dict, which will handle the mapping
        return super().from_yaml(file_path, **kwargs)

    @classmethod
    def from_existing(cls, name: str, **kwargs) -> "LLMRouter":  # type: ignore[override]
        """Load from existing with backward compatibility."""
        # The parent from_existing calls from_dict, which will handle the mapping
        return super().from_existing(name, **kwargs)  # type: ignore[return-value]

    def export_with_embeddings(self, file_path: str):
        """Export with embeddings using 'tiers' for backward compatibility."""
        import json
        from pathlib import Path

        from redisvl.extensions.router.schema import (
            PretrainedReference,
            PretrainedRoute,
        )
        from redisvl.utils.utils import model_to_dict

        fp = Path(file_path).resolve()

        pretrained_tiers = []
        for route in self.routes:
            # Get embeddings for all references
            vectors = self.vectorizer.embed_many(route.references)

            references = [
                PretrainedReference(text=ref, vector=vec)
                for ref, vec in zip(route.references, vectors)
            ]

            pretrained_tiers.append(
                PretrainedRoute(
                    name=route.name,
                    references=references,
                    metadata=route.metadata,
                    distance_threshold=route.distance_threshold,
                    model=route.model,
                )
            )

        # Use "tiers" instead of "routes" for backward compatibility
        config_dict = {
            "name": self.name,
            "version": "1.0.0",
            "vectorizer": {
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            "tiers": [model_to_dict(tier) for tier in pretrained_tiers],
            "routing_config": model_to_dict(self.routing_config),
        }

        with open(fp, "w") as f:
            json.dump(config_dict, f, indent=2)


# Backward compatibility wrapper for AsyncLLMRouter
class AsyncLLMRouter(_AsyncSemanticRouter):
    """Backward compatibility wrapper for AsyncLLMRouter."""

    @classmethod
    async def create(  # type: ignore[override]
        cls,
        name: str,
        tiers: Optional[List[Route]] = None,
        routes: Optional[List[Route]] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        routing_config: Optional[Any] = None,
        redis_client: Optional[AsyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        cost_optimization: bool = False,
        connection_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "AsyncLLMRouter":
        """Create AsyncLLMRouter (deprecated, use AsyncSemanticRouter.create).

        Args:
            name: Router name.
            tiers: (Deprecated) Use routes instead.
            routes: List of Route objects.
            vectorizer: Vectorizer for embeddings.
            routing_config: Configuration for routing behavior.
            redis_client: Async Redis client.
            redis_url: Redis URL.
            overwrite: Whether to overwrite existing index.
            cost_optimization: Enable cost-aware routing.
            connection_kwargs: Additional Redis connection arguments.
        """
        # Map tiers → routes for backward compatibility
        if tiers is not None and routes is None:
            routes = tiers
        elif routes is None:
            routes = []

        # Map cost_optimization to routing_config
        if cost_optimization and routing_config:
            routing_config.cost_optimization = True
        elif cost_optimization:
            from redisvl.extensions.router.schema import RoutingConfig

            routing_config = RoutingConfig(cost_optimization=True)

        # Create the async semantic router
        router = await _AsyncSemanticRouter.create(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
            redis_client=redis_client,
            redis_url=redis_url,
            overwrite=overwrite,
            connection_kwargs=connection_kwargs,
        )

        # Wrap in AsyncLLMRouter for backward compatibility
        async_llm_router = cls.model_construct(
            name=router.name,
            routes=router.routes,
            vectorizer=router.vectorizer,
            routing_config=router.routing_config,
        )
        # Copy the index
        object.__setattr__(async_llm_router, "_index", router._index)
        return async_llm_router

    async def route(self, query: Optional[str] = None, **kwargs):
        """Route a query (deprecated method, use __call__ instead)."""
        return await self(statement=query, **kwargs)

    @property
    def tiers(self):
        """Alias for routes (backward compatibility)."""
        return self.routes

    @tiers.setter
    def tiers(self, value):
        """Alias for routes (backward compatibility)."""
        self.routes = value

    @property
    def tier_names(self):
        """Alias for route_names (backward compatibility)."""
        return self.route_names

    @property
    def tier_thresholds(self):
        """Alias for route_thresholds (backward compatibility)."""
        return self.route_thresholds

    @property
    def default_tier(self):
        """Alias for default_route (backward compatibility)."""
        return self.routing_config.default_route

    def get_tier(self, tier_name: str):
        """Alias for get (backward compatibility)."""
        return self.get(tier_name)

    async def add_tier(self, tier: Route):
        """Add a new tier (backward compatibility)."""
        if self.get(tier.name):
            raise ValueError(f"Tier {tier.name} already exists")
        await self._add_routes([tier])
        await self._update_router_state()

    async def remove_tier(self, tier_name: str):
        """Remove a tier (backward compatibility)."""
        return await self.remove_route(tier_name)

    async def add_tier_references(self, tier_name: str, references):
        """Add references to a tier (backward compatibility)."""
        return await self.add_route_references(tier_name, references)

    async def update_tier_threshold(self, tier_name: str, threshold: float):
        """Update a tier's distance threshold (backward compatibility)."""
        route = self.get(tier_name)
        if route is None:
            raise ValueError(f"Tier {tier_name} not found")
        if not (0 < threshold <= 2):
            raise ValueError("Threshold must be in range (0, 2]")
        route.distance_threshold = threshold  # type: ignore
        await self._update_router_state()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with 'tiers' for backward compatibility."""
        result = super().to_dict()
        # Map routes → tiers for backward compatibility
        result["tiers"] = result.pop("routes")
        return result

    @classmethod
    async def from_dict(cls, data: Dict[str, Any], **kwargs):
        """Load from dict with 'tiers' for backward compatibility."""
        # Map tiers → routes for backward compatibility
        if "tiers" in data and "routes" not in data:
            data = data.copy()
            data["routes"] = data.pop("tiers")
        router = await _AsyncSemanticRouter.from_dict(data, **kwargs)

        # Wrap in AsyncLLMRouter for backward compatibility
        async_llm_router = cls.model_construct(
            name=router.name,
            routes=router.routes,
            vectorizer=router.vectorizer,
            routing_config=router.routing_config,
        )
        object.__setattr__(async_llm_router, "_index", router._index)
        return async_llm_router

    @classmethod
    async def from_yaml(cls, file_path: str, **kwargs):
        """Load from YAML with 'tiers' for backward compatibility."""
        from pathlib import Path

        import yaml

        fp = Path(file_path).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        with open(fp, "r") as f:
            yaml_data = yaml.safe_load(f)

        # Use our from_dict which handles tiers → routes mapping
        return await cls.from_dict(yaml_data, **kwargs)

    @classmethod
    async def from_existing(cls, name: str, **kwargs) -> "AsyncLLMRouter":  # type: ignore[override]
        """Load from existing with backward compatibility."""
        router = await _AsyncSemanticRouter.from_existing(name, **kwargs)

        # Wrap in AsyncLLMRouter for backward compatibility
        async_llm_router = cls.model_construct(
            name=router.name,
            routes=router.routes,
            vectorizer=router.vectorizer,
            routing_config=router.routing_config,
        )
        object.__setattr__(async_llm_router, "_index", router._index)
        return async_llm_router

    async def export_with_embeddings(self, file_path: str):
        """Export with embeddings using 'tiers' for backward compatibility (async)."""
        import json
        from pathlib import Path

        from redisvl.extensions.router.schema import (
            PretrainedReference,
            PretrainedRoute,
        )
        from redisvl.utils.utils import model_to_dict

        fp = Path(file_path).resolve()

        pretrained_tiers = []
        for route in self.routes:
            # Get embeddings for all references (use async method)
            vectors = await self.vectorizer.aembed_many(route.references)

            references = [
                PretrainedReference(text=ref, vector=vec)
                for ref, vec in zip(route.references, vectors)
            ]

            pretrained_tiers.append(
                PretrainedRoute(
                    name=route.name,
                    references=references,
                    metadata=route.metadata,
                    distance_threshold=route.distance_threshold,
                    model=route.model,
                )
            )

        # Use "tiers" instead of "routes" for backward compatibility
        config_dict = {
            "name": self.name,
            "version": "1.0.0",
            "vectorizer": {
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            "tiers": [model_to_dict(tier) for tier in pretrained_tiers],
            "routing_config": model_to_dict(self.routing_config),
        }

        with open(fp, "w") as f:
            json.dump(config_dict, f, indent=2)


# Backward compatibility aliases
ModelTier = Route
LLMRouteMatch = RouteMatch
PretrainedTier = PretrainedRoute

__all__ = [
    "AsyncLLMRouter",
    "LLMRouter",
    "ModelTier",
    "LLMRouteMatch",
    "DistanceAggregationMethod",
    "PretrainedReference",
    "PretrainedTier",
    "PretrainedRouterConfig",
]
