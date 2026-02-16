"""LLM Router implementation.

Intelligent LLM model selection using semantic routing. Routes queries to the
most appropriate model tier based on semantic similarity to reference phrases.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import redis.commands.search.reducers as reducers
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from redis.commands.search.aggregation import AggregateRequest, AggregateResult, Reducer
from redis.exceptions import ResponseError

from redisvl.extensions.constants import ROUTE_VECTOR_FIELD_NAME
from redisvl.extensions.llm_router.schema import (
    DistanceAggregationMethod,
    LLMRouteMatch,
    ModelTier,
    PretrainedReference,
    PretrainedRouterConfig,
    PretrainedTier,
    RoutingConfig,
)
from redisvl.extensions.router.schema import Route, SemanticRouterIndexSchema
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import VectorRangeQuery
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.redis.utils import convert_bytes, hashify, make_dict
from redisvl.types import AsyncRedisClient, SyncRedisClient
from redisvl.utils.log import get_logger
from redisvl.utils.utils import model_to_dict, scan_by_pattern
from redisvl.utils.vectorize.base import BaseVectorizer
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

logger = get_logger(__name__)


class LLMRouter(BaseModel):
    """Intelligent LLM Router for model tier selection.

    Routes queries to the most appropriate LLM model based on semantic similarity
    to reference phrases. Uses Redis vector search for fast, scalable routing.

    Example:
        >>> from redisvl.extensions.llm_router import LLMRouter, ModelTier
        >>>
        >>> tiers = [
        ...     ModelTier(
        ...         name="simple",
        ...         model="openai/gpt-4.1-nano",
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
        >>> print(match.tier, match.model)
        simple openai/gpt-4.1-nano
    """

    name: str
    """Router name (also used as Redis index prefix)."""

    tiers: List[ModelTier]
    """List of model tiers for routing."""

    vectorizer: BaseVectorizer = Field(default_factory=HFTextVectorizer)
    """Vectorizer for embedding queries and references."""

    routing_config: RoutingConfig = Field(default_factory=RoutingConfig)
    """Configuration for routing behavior."""

    _index: SearchIndex = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str,
        tiers: List[ModelTier],
        vectorizer: Optional[BaseVectorizer] = None,
        routing_config: Optional[RoutingConfig] = None,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        cost_optimization: bool = False,
        connection_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """Initialize the LLMRouter.

        Args:
            name: Router name (used as Redis index prefix).
            tiers: List of ModelTier objects defining routing targets.
            vectorizer: Vectorizer for embeddings. Defaults to HFTextVectorizer.
            routing_config: Configuration for routing behavior.
            redis_client: Existing Redis client. Defaults to None.
            redis_url: Redis connection URL. Defaults to "redis://localhost:6379".
            overwrite: Whether to overwrite existing index. Defaults to False.
            cost_optimization: Enable cost-aware routing. Defaults to False.
            connection_kwargs: Additional Redis connection arguments.
        """
        # Set up vectorizer
        if vectorizer is None:
            vectorizer = HFTextVectorizer(
                model="sentence-transformers/all-mpnet-base-v2"
            )
        elif not isinstance(vectorizer, BaseVectorizer):
            raise TypeError("Must provide a valid redisvl.vectorizer class.")

        # Set up routing config
        if routing_config is None:
            routing_config = RoutingConfig(cost_optimization=cost_optimization)
        elif cost_optimization:
            routing_config.cost_optimization = True

        super().__init__(
            name=name,
            tiers=tiers,
            vectorizer=vectorizer,
            routing_config=routing_config,
        )

        self._initialize_index(redis_client, redis_url, overwrite, **connection_kwargs)

        # Store router config in Redis
        self._index.client.json().set(f"{self.name}:router_config", ".", self.to_dict())  # type: ignore

    @classmethod
    def from_existing(
        cls,
        name: str,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        **kwargs,
    ) -> "LLMRouter":
        """Reconnect to an existing LLMRouter.

        Args:
            name: Router name.
            redis_client: Existing Redis client.
            redis_url: Redis connection URL.

        Returns:
            LLMRouter instance connected to existing index.
        """
        if redis_client:
            RedisConnectionFactory.validate_sync_redis(redis_client)
        elif redis_url:
            redis_client = RedisConnectionFactory.get_redis_connection(
                redis_url=redis_url, **kwargs
            )

        if redis_client is None:
            raise ValueError("Could not establish Redis connection.")

        router_dict = redis_client.json().get(f"{name}:router_config")
        if not isinstance(router_dict, dict):
            raise ValueError(f"No router config found for {name}")

        return cls.from_dict(
            router_dict, redis_url=redis_url, redis_client=redis_client
        )

    def _initialize_index(
        self,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **connection_kwargs,
    ):
        """Initialize the Redis search index."""
        schema = SemanticRouterIndexSchema.from_params(
            self.name, self.vectorizer.dims, self.vectorizer.dtype  # type: ignore
        )

        self._index = SearchIndex(
            schema=schema,
            redis_client=redis_client,
            redis_url=redis_url,
            **connection_kwargs,
        )

        existed = self._index.exists()
        if not overwrite and existed:
            existing_index = SearchIndex.from_existing(
                self.name, redis_client=self._index.client
            )
            if existing_index.schema.to_dict() != self._index.schema.to_dict():
                raise ValueError(
                    f"Existing index {self.name} schema does not match. "
                    "Set overwrite=True to recreate."
                )

        self._index.create(overwrite=overwrite, drop=False)

        if not existed or overwrite:
            self._add_tiers(self.tiers)

    def _add_tiers(self, tiers: List[ModelTier]):
        """Add tiers to the router index."""
        tier_references: List[Dict[str, Any]] = []
        keys: List[str] = []

        for tier in tiers:
            # Embed all references for this tier
            reference_vectors = self.vectorizer.embed_many(
                tier.references, as_buffer=True
            )

            for i, reference in enumerate(tier.references):
                reference_hash = hashify(reference)
                tier_references.append(
                    {
                        "reference_id": reference_hash,
                        "route_name": tier.name,  # Use route_name for compatibility
                        "reference": reference,
                        "vector": reference_vectors[i],
                    }
                )
                keys.append(self._tier_ref_key(tier.name, reference_hash))

            # Add tier to local list if not present
            if not self.get_tier(tier.name):
                self.tiers.append(tier)

        self._index.load(tier_references, keys=keys)

    def _tier_ref_key(self, tier_name: str, reference_hash: str) -> str:
        """Generate key for a tier reference."""
        sep = self._index.key_separator
        prefix = (
            self._index.prefix.rstrip(sep)
            if sep and self._index.prefix
            else self._index.prefix
        )
        if prefix:
            return f"{prefix}{sep}{tier_name}{sep}{reference_hash}"
        return f"{tier_name}{sep}{reference_hash}"

    def _tier_pattern(self, tier_name: str) -> str:
        """Generate search pattern for tier references."""
        sep = self._index.key_separator
        prefix = (
            self._index.prefix.rstrip(sep)
            if sep and self._index.prefix
            else self._index.prefix
        )
        if prefix:
            return f"{prefix}{sep}{tier_name}{sep}*"
        return f"{tier_name}{sep}*"

    @property
    def tier_names(self) -> List[str]:
        """Get list of tier names."""
        return [tier.name for tier in self.tiers]

    @property
    def tier_thresholds(self) -> Dict[str, float]:
        """Get distance thresholds for each tier."""
        return {tier.name: tier.distance_threshold for tier in self.tiers}

    @property
    def default_tier(self) -> Optional[str]:
        """Get default tier name."""
        return self.routing_config.default_tier

    def get_tier(self, tier_name: str) -> Optional[ModelTier]:
        """Get a tier by name."""
        return next((t for t in self.tiers if t.name == tier_name), None)

    def add_tier(self, tier: ModelTier):
        """Add a new tier to the router."""
        if self.get_tier(tier.name):
            raise ValueError(f"Tier {tier.name} already exists")
        self._add_tiers([tier])
        self._update_router_config()

    def remove_tier(self, tier_name: str):
        """Remove a tier from the router."""
        tier = self.get_tier(tier_name)
        if tier is None:
            logger.warning(f"Tier {tier_name} not found")
            return

        # Delete references from index
        pattern = self._tier_pattern(tier_name)
        keys = scan_by_pattern(self._index.client, pattern)  # type: ignore
        if keys:
            self._index.drop_keys(list(keys))

        # Remove from local list
        self.tiers = [t for t in self.tiers if t.name != tier_name]
        self._update_router_config()

    def add_tier_references(
        self,
        tier_name: str,
        references: Union[str, List[str]],
    ):
        """Add references to an existing tier."""
        tier = self.get_tier(tier_name)
        if tier is None:
            raise ValueError(f"Tier {tier_name} not found")

        if isinstance(references, str):
            references = [references]

        # Embed and add references
        reference_vectors = self.vectorizer.embed_many(references, as_buffer=True)
        tier_references = []
        keys = []

        for i, reference in enumerate(references):
            reference_hash = hashify(reference)
            tier_references.append(
                {
                    "reference_id": reference_hash,
                    "route_name": tier_name,
                    "reference": reference,
                    "vector": reference_vectors[i],
                }
            )
            keys.append(self._tier_ref_key(tier_name, reference_hash))

        self._index.load(tier_references, keys=keys)
        tier.references.extend(references)
        self._update_router_config()

    def update_tier_threshold(self, tier_name: str, threshold: float):
        """Update a tier's distance threshold."""
        tier = self.get_tier(tier_name)
        if tier is None:
            raise ValueError(f"Tier {tier_name} not found")
        tier.distance_threshold = threshold
        self._update_router_config()

    def _distance_threshold_filter(self) -> str:
        """Build filter expression for per-tier thresholds."""
        filters = []
        for tier in self.tiers:
            filters.append(
                f"(@route_name == '{tier.name}' && @distance < {tier.distance_threshold})"
            )
        return " || ".join(filters)

    def _build_aggregate_request(
        self,
        vector_range_query: VectorRangeQuery,
        aggregation_method: DistanceAggregationMethod,
        max_k: int,
    ) -> AggregateRequest:
        """Build Redis aggregation request."""
        if aggregation_method == DistanceAggregationMethod.min:
            agg_func = reducers.min
        elif aggregation_method == DistanceAggregationMethod.sum:
            agg_func = reducers.sum  # type: ignore
        else:
            agg_func = reducers.avg  # type: ignore

        query_str = str(vector_range_query).split(" RETURN")[0]
        request = (
            AggregateRequest(query_str)
            .group_by("@route_name", agg_func("vector_distance").alias("distance"))
            .sort_by("@distance", max=max_k)
            .dialect(2)
        )
        request.filter(self._distance_threshold_filter())
        return request

    def _get_tier_matches(
        self,
        vector: List[float],
        aggregation_method: DistanceAggregationMethod,
        max_k: int = 1,
    ) -> List[LLMRouteMatch]:
        """Get matching tiers for a vector."""
        distance_threshold = max(t.distance_threshold for t in self.tiers)

        query = VectorRangeQuery(
            vector=vector,
            vector_field_name=ROUTE_VECTOR_FIELD_NAME,
            distance_threshold=float(distance_threshold),
            return_fields=["route_name"],
        )

        request = self._build_aggregate_request(query, aggregation_method, max_k)

        try:
            result = self._index.aggregate(request, query.params)
        except ResponseError as e:
            if "VSS is not yet supported on FT.AGGREGATE" in str(e):
                raise RuntimeError("LLM routing requires Redis 7.x or greater")
            raise

        matches = []
        for row in result.rows:
            row_dict = make_dict(convert_bytes(row))
            tier_name = row_dict["route_name"]
            tier = self.get_tier(tier_name)
            distance = float(row_dict["distance"])

            matches.append(
                LLMRouteMatch(
                    tier=tier_name,
                    model=tier.model if tier else None,
                    distance=distance,
                    confidence=1 - (distance / 2),  # Convert to 0-1 range
                    metadata=tier.metadata if tier else {},
                )
            )

        return matches

    def _apply_cost_optimization(
        self, matches: List[LLMRouteMatch]
    ) -> List[LLMRouteMatch]:
        """Re-rank matches considering cost."""
        if not matches or not self.routing_config.cost_optimization:
            return matches

        cost_weight = self.routing_config.cost_weight
        ranked = []

        for match in matches:
            cost = match.metadata.get("cost_per_1k_input", 0)
            # Add cost penalty to distance (normalized)
            adjusted_distance = match.distance + (cost * cost_weight)
            ranked.append((match, adjusted_distance))

        ranked.sort(key=lambda x: x[1])
        return [m for m, _ in ranked]

    def route(
        self,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        aggregation_method: Optional[DistanceAggregationMethod] = None,
    ) -> LLMRouteMatch:
        """Route a query to the best matching tier.

        Args:
            query: Text query to route.
            vector: Pre-computed embedding vector.
            aggregation_method: Override default aggregation method.

        Returns:
            LLMRouteMatch with tier, model, and routing metadata.
        """
        if vector is None:
            if query is None:
                raise ValueError("Must provide query or vector")
            vector = self.vectorizer.embed(query)

        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        matches = self._get_tier_matches(
            vector, aggregation_method, max_k=len(self.tiers)
        )
        matches = self._apply_cost_optimization(matches)

        if not matches:
            # Return default tier or empty match
            if self.default_tier:
                tier = self.get_tier(self.default_tier)
                if tier:
                    return LLMRouteMatch(
                        tier=tier.name,
                        model=tier.model,
                        metadata=tier.metadata,
                    )
            return LLMRouteMatch()

        top_match = matches[0]
        top_match.alternatives = [(m.tier, m.distance) for m in matches[1:]]
        return top_match

    def route_many(
        self,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        max_k: Optional[int] = None,
        aggregation_method: Optional[DistanceAggregationMethod] = None,
    ) -> List[LLMRouteMatch]:
        """Route a query and return multiple tier matches.

        Args:
            query: Text query to route.
            vector: Pre-computed embedding vector.
            max_k: Maximum number of matches to return.
            aggregation_method: Override default aggregation method.

        Returns:
            List of LLMRouteMatch objects ordered by distance.
        """
        if vector is None:
            if query is None:
                raise ValueError("Must provide query or vector")
            vector = self.vectorizer.embed(query)

        max_k = max_k or self.routing_config.max_k
        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        matches = self._get_tier_matches(vector, aggregation_method, max_k)
        return self._apply_cost_optimization(matches)

    def __call__(
        self,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
    ) -> LLMRouteMatch:
        """Shorthand for route()."""
        return self.route(query=query, vector=vector)

    # Serialization methods

    def to_dict(self) -> Dict[str, Any]:
        """Serialize router to dictionary."""
        return {
            "name": self.name,
            "tiers": [model_to_dict(tier) for tier in self.tiers],
            "vectorizer": {
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            "routing_config": model_to_dict(self.routing_config),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        **kwargs,
    ) -> "LLMRouter":
        """Create router from dictionary."""
        from redisvl.utils.vectorize import vectorizer_from_dict

        try:
            name = data["name"]
            tiers_data = data["tiers"]
            vectorizer_data = data["vectorizer"]
            routing_config_data = data.get("routing_config", {})
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")

        vectorizer = vectorizer_from_dict(vectorizer_data)
        if not vectorizer:
            raise ValueError(f"Could not load vectorizer: {vectorizer_data}")

        tiers = [ModelTier(**t) for t in tiers_data]
        routing_config = RoutingConfig(**routing_config_data)

        return cls(
            name=name,
            tiers=tiers,
            vectorizer=vectorizer,
            routing_config=routing_config,
            **kwargs,
        )

    def to_yaml(self, file_path: str, overwrite: bool = True):
        """Save router to YAML file."""
        fp = Path(file_path).resolve()
        if fp.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists")

        with open(fp, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, file_path: str, **kwargs) -> "LLMRouter":
        """Load router from YAML file."""
        fp = Path(file_path).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        with open(fp) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data, **kwargs)

    def export_with_embeddings(self, file_path: str):
        """Export router with pre-computed embeddings.

        This allows loading the router without re-embedding references.

        Args:
            file_path: Path to JSON file.
        """
        fp = Path(file_path).resolve()

        pretrained_tiers = []
        for tier in self.tiers:
            # Get embeddings for all references
            vectors = self.vectorizer.embed_many(tier.references)

            references = [
                PretrainedReference(text=ref, vector=vec)
                for ref, vec in zip(tier.references, vectors)
            ]

            pretrained_tiers.append(
                PretrainedTier(
                    name=tier.name,
                    model=tier.model,
                    references=references,
                    metadata=tier.metadata,
                    distance_threshold=tier.distance_threshold,
                )
            )

        config = PretrainedRouterConfig(
            name=self.name,
            vectorizer={
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            tiers=pretrained_tiers,
            routing_config=model_to_dict(self.routing_config),
        )

        with open(fp, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        config_name_or_path: str,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        **kwargs,
    ) -> "LLMRouter":
        """Load router from pretrained config with embeddings.

        This skips the embedding step by using pre-computed vectors.
        Accepts either a file path or a built-in config name (e.g., "default").

        Args:
            config_name_or_path: Path to pretrained JSON file, or name of a
                built-in config (e.g., "default").
            redis_client: Redis client.
            redis_url: Redis URL.

        Returns:
            LLMRouter loaded without re-embedding.
        """
        import numpy as np

        from redisvl.utils.vectorize import vectorizer_from_dict

        fp = Path(config_name_or_path)
        if not fp.exists():
            from redisvl.extensions.llm_router.pretrained import get_pretrained_path

            fp = get_pretrained_path(config_name_or_path)

        with open(fp) as f:
            data = json.load(f)

        config = PretrainedRouterConfig(**data)

        # Create vectorizer (for future queries, not for loading)
        vectorizer = vectorizer_from_dict(config.vectorizer)
        if not vectorizer:
            raise ValueError(f"Could not load vectorizer: {config.vectorizer}")

        # Set up connection — prefer provided client over URL
        if redis_client:
            RedisConnectionFactory.validate_sync_redis(redis_client)
        elif redis_url:
            redis_client = RedisConnectionFactory.get_redis_connection(
                redis_url=redis_url, **kwargs
            )

        if redis_client is None:
            raise ValueError("Could not establish Redis connection")

        # Create index schema
        schema = SemanticRouterIndexSchema.from_params(
            config.name, vectorizer.dims, vectorizer.dtype  # type: ignore
        )

        index = SearchIndex(
            schema=schema,
            redis_client=redis_client,
        )
        index.create(overwrite=True, drop=False)

        # Load pre-computed embeddings directly
        tiers = []
        all_references = []
        all_keys = []

        for pt in config.tiers:
            tier = ModelTier(
                name=pt.name,
                model=pt.model,
                references=[r.text for r in pt.references],
                metadata=pt.metadata,
                distance_threshold=pt.distance_threshold,
            )
            tiers.append(tier)

            # Use pre-computed vectors (convert to buffer)
            for ref in pt.references:
                vector_buffer = np.array(ref.vector, dtype=np.float32).tobytes()
                reference_hash = hashify(ref.text)

                sep = index.key_separator
                prefix = (
                    index.prefix.rstrip(sep) if sep and index.prefix else index.prefix
                )
                if prefix:
                    key = f"{prefix}{sep}{pt.name}{sep}{reference_hash}"
                else:
                    key = f"{pt.name}{sep}{reference_hash}"

                all_references.append(
                    {
                        "reference_id": reference_hash,
                        "route_name": pt.name,
                        "reference": ref.text,
                        "vector": vector_buffer,
                    }
                )
                all_keys.append(key)

        index.load(all_references, keys=all_keys)

        # Create router instance using Pydantic's model_construct to bypass __init__
        routing_config = RoutingConfig(**config.routing_config)

        router = cls.model_construct(
            name=config.name,
            tiers=tiers,
            vectorizer=vectorizer,
            routing_config=routing_config,
        )
        # Set private attribute directly
        object.__setattr__(router, "_index", index)

        # Store config
        redis_client.json().set(f"{config.name}:router_config", ".", router.to_dict())

        return router

    def _update_router_config(self):
        """Update router config in Redis."""
        self._index.client.json().set(f"{self.name}:router_config", ".", self.to_dict())

    def delete(self):
        """Delete the router index."""
        self._index.client.delete(f"{self.name}:router_config")
        self._index.delete(drop=True)

    def clear(self):
        """Clear all tier references."""
        self._index.clear()
        self.tiers = []


class AsyncLLMRouter(BaseModel):
    """Async LLM Router for model tier selection.

    Provides the same functionality as :class:`LLMRouter` but uses async I/O.
    Since ``__init__`` cannot be async, use the :meth:`create` classmethod
    factory to instantiate.

    Example:
        >>> from redisvl.extensions.llm_router import AsyncLLMRouter, ModelTier
        >>>
        >>> tiers = [
        ...     ModelTier(
        ...         name="simple",
        ...         model="openai/gpt-4.1-nano",
        ...         references=["hello", "hi", "thanks"],
        ...         distance_threshold=0.5,
        ...     ),
        ... ]
        >>>
        >>> router = await AsyncLLMRouter.create(
        ...     name="my-router",
        ...     tiers=tiers,
        ...     redis_url="redis://localhost:6379",
        ... )
        >>>
        >>> match = await router.route("hello, how are you?")
        >>> print(match.tier, match.model)
    """

    name: str
    """Router name (also used as Redis index prefix)."""

    tiers: List[ModelTier]
    """List of model tiers for routing."""

    vectorizer: BaseVectorizer = Field(default_factory=HFTextVectorizer)
    """Vectorizer for embedding queries and references."""

    routing_config: RoutingConfig = Field(default_factory=RoutingConfig)
    """Configuration for routing behavior."""

    _index: AsyncSearchIndex = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    async def create(
        cls,
        name: str,
        tiers: List[ModelTier],
        vectorizer: Optional[BaseVectorizer] = None,
        routing_config: Optional[RoutingConfig] = None,
        redis_client: Optional[AsyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        cost_optimization: bool = False,
        connection_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> "AsyncLLMRouter":
        """Create an AsyncLLMRouter instance (async factory).

        Args:
            name: Router name (used as Redis index prefix).
            tiers: List of ModelTier objects defining routing targets.
            vectorizer: Vectorizer for embeddings. Defaults to HFTextVectorizer.
            routing_config: Configuration for routing behavior.
            redis_client: Existing async Redis client.
            redis_url: Redis connection URL.
            overwrite: Whether to overwrite existing index.
            cost_optimization: Enable cost-aware routing.
            connection_kwargs: Additional Redis connection arguments.

        Returns:
            AsyncLLMRouter instance.
        """
        if vectorizer is None:
            vectorizer = HFTextVectorizer(
                model="sentence-transformers/all-mpnet-base-v2"
            )
        elif not isinstance(vectorizer, BaseVectorizer):
            raise TypeError("Must provide a valid redisvl.vectorizer class.")

        if routing_config is None:
            routing_config = RoutingConfig(cost_optimization=cost_optimization)
        elif cost_optimization:
            routing_config.cost_optimization = True

        router = cls.model_construct(
            name=name,
            tiers=tiers,
            vectorizer=vectorizer,
            routing_config=routing_config,
        )

        await router._initialize_index(
            redis_client, redis_url, overwrite, **connection_kwargs
        )

        client = router._index.client
        await client.json().set(f"{name}:router_config", ".", router.to_dict())  # type: ignore
        return router

    @classmethod
    async def from_existing(
        cls,
        name: str,
        redis_client: Optional[AsyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        **kwargs,
    ) -> "AsyncLLMRouter":
        """Reconnect to an existing AsyncLLMRouter.

        Args:
            name: Router name.
            redis_client: Existing async Redis client.
            redis_url: Redis connection URL.

        Returns:
            AsyncLLMRouter instance connected to existing index.
        """
        if redis_client:
            await RedisConnectionFactory.validate_async_redis(redis_client)
        elif redis_url:
            redis_client = await RedisConnectionFactory._get_aredis_connection(
                redis_url=redis_url, **kwargs
            )

        if redis_client is None:
            raise ValueError("Could not establish Redis connection.")

        router_dict = await redis_client.json().get(f"{name}:router_config")  # type: ignore
        if not isinstance(router_dict, dict):
            raise ValueError(f"No router config found for {name}")

        return await cls.from_dict(
            router_dict, redis_url=redis_url, redis_client=redis_client
        )

    async def _initialize_index(
        self,
        redis_client: Optional[AsyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **connection_kwargs,
    ):
        """Initialize the Redis search index (async)."""
        schema = SemanticRouterIndexSchema.from_params(
            self.name, self.vectorizer.dims, self.vectorizer.dtype  # type: ignore
        )

        self._index = AsyncSearchIndex(
            schema=schema,
            redis_client=redis_client,
            redis_url=redis_url,
            **connection_kwargs,
        )

        existed = await self._index.exists()
        if not overwrite and existed:
            existing_index = await AsyncSearchIndex.from_existing(
                self.name, redis_client=self._index.client
            )
            if existing_index.schema.to_dict() != self._index.schema.to_dict():
                raise ValueError(
                    f"Existing index {self.name} schema does not match. "
                    "Set overwrite=True to recreate."
                )

        await self._index.create(overwrite=overwrite, drop=False)

        if not existed or overwrite:
            await self._add_tiers(self.tiers)

    async def _add_tiers(self, tiers: List[ModelTier]):
        """Add tiers to the router index (async)."""
        tier_references: List[Dict[str, Any]] = []
        keys: List[str] = []

        for tier in tiers:
            reference_vectors = await self.vectorizer.aembed_many(
                tier.references, as_buffer=True
            )

            for i, reference in enumerate(tier.references):
                reference_hash = hashify(reference)
                tier_references.append(
                    {
                        "reference_id": reference_hash,
                        "route_name": tier.name,
                        "reference": reference,
                        "vector": reference_vectors[i],
                    }
                )
                keys.append(self._tier_ref_key(tier.name, reference_hash))

            if not self.get_tier(tier.name):
                self.tiers.append(tier)

        await self._index.load(tier_references, keys=keys)

    def _tier_ref_key(self, tier_name: str, reference_hash: str) -> str:
        """Generate key for a tier reference."""
        sep = self._index.key_separator
        prefix = (
            self._index.prefix.rstrip(sep)
            if sep and self._index.prefix
            else self._index.prefix
        )
        if prefix:
            return f"{prefix}{sep}{tier_name}{sep}{reference_hash}"
        return f"{tier_name}{sep}{reference_hash}"

    def _tier_pattern(self, tier_name: str) -> str:
        """Generate search pattern for tier references."""
        sep = self._index.key_separator
        prefix = (
            self._index.prefix.rstrip(sep)
            if sep and self._index.prefix
            else self._index.prefix
        )
        if prefix:
            return f"{prefix}{sep}{tier_name}{sep}*"
        return f"{tier_name}{sep}*"

    @property
    def tier_names(self) -> List[str]:
        """Get list of tier names."""
        return [tier.name for tier in self.tiers]

    @property
    def tier_thresholds(self) -> Dict[str, float]:
        """Get distance thresholds for each tier."""
        return {tier.name: tier.distance_threshold for tier in self.tiers}

    @property
    def default_tier(self) -> Optional[str]:
        """Get default tier name."""
        return self.routing_config.default_tier

    def get_tier(self, tier_name: str) -> Optional[ModelTier]:
        """Get a tier by name."""
        return next((t for t in self.tiers if t.name == tier_name), None)

    async def add_tier(self, tier: ModelTier):
        """Add a new tier to the router."""
        if self.get_tier(tier.name):
            raise ValueError(f"Tier {tier.name} already exists")
        await self._add_tiers([tier])
        await self._update_router_config()

    async def remove_tier(self, tier_name: str):
        """Remove a tier from the router."""
        tier = self.get_tier(tier_name)
        if tier is None:
            logger.warning(f"Tier {tier_name} not found")
            return

        pattern = self._tier_pattern(tier_name)
        client = self._index.client
        keys = []
        async for key in client.scan_iter(match=pattern):  # type: ignore
            keys.append(key)
        if keys:
            await self._index.drop_keys(keys)

        self.tiers = [t for t in self.tiers if t.name != tier_name]
        await self._update_router_config()

    async def add_tier_references(
        self,
        tier_name: str,
        references: Union[str, List[str]],
    ):
        """Add references to an existing tier."""
        tier = self.get_tier(tier_name)
        if tier is None:
            raise ValueError(f"Tier {tier_name} not found")

        if isinstance(references, str):
            references = [references]

        reference_vectors = await self.vectorizer.aembed_many(
            references, as_buffer=True
        )
        tier_references = []
        keys = []

        for i, reference in enumerate(references):
            reference_hash = hashify(reference)
            tier_references.append(
                {
                    "reference_id": reference_hash,
                    "route_name": tier_name,
                    "reference": reference,
                    "vector": reference_vectors[i],
                }
            )
            keys.append(self._tier_ref_key(tier_name, reference_hash))

        await self._index.load(tier_references, keys=keys)
        tier.references.extend(references)
        await self._update_router_config()

    async def update_tier_threshold(self, tier_name: str, threshold: float):
        """Update a tier's distance threshold."""
        tier = self.get_tier(tier_name)
        if tier is None:
            raise ValueError(f"Tier {tier_name} not found")
        tier.distance_threshold = threshold
        await self._update_router_config()

    def _distance_threshold_filter(self) -> str:
        """Build filter expression for per-tier thresholds."""
        filters = []
        for tier in self.tiers:
            filters.append(
                f"(@route_name == '{tier.name}' && @distance < {tier.distance_threshold})"
            )
        return " || ".join(filters)

    def _build_aggregate_request(
        self,
        vector_range_query: VectorRangeQuery,
        aggregation_method: DistanceAggregationMethod,
        max_k: int,
    ) -> AggregateRequest:
        """Build Redis aggregation request."""
        if aggregation_method == DistanceAggregationMethod.min:
            agg_func = reducers.min
        elif aggregation_method == DistanceAggregationMethod.sum:
            agg_func = reducers.sum  # type: ignore
        else:
            agg_func = reducers.avg  # type: ignore

        query_str = str(vector_range_query).split(" RETURN")[0]
        request = (
            AggregateRequest(query_str)
            .group_by("@route_name", agg_func("vector_distance").alias("distance"))
            .sort_by("@distance", max=max_k)
            .dialect(2)
        )
        request.filter(self._distance_threshold_filter())
        return request

    async def _get_tier_matches(
        self,
        vector: List[float],
        aggregation_method: DistanceAggregationMethod,
        max_k: int = 1,
    ) -> List[LLMRouteMatch]:
        """Get matching tiers for a vector (async)."""
        distance_threshold = max(t.distance_threshold for t in self.tiers)

        query = VectorRangeQuery(
            vector=vector,
            vector_field_name=ROUTE_VECTOR_FIELD_NAME,
            distance_threshold=float(distance_threshold),
            return_fields=["route_name"],
        )

        request = self._build_aggregate_request(query, aggregation_method, max_k)

        try:
            result = await self._index.aggregate(request, query.params)
        except ResponseError as e:
            if "VSS is not yet supported on FT.AGGREGATE" in str(e):
                raise RuntimeError("LLM routing requires Redis 7.x or greater")
            raise

        matches = []
        for row in result.rows:
            row_dict = make_dict(convert_bytes(row))
            tier_name = row_dict["route_name"]
            tier = self.get_tier(tier_name)
            distance = float(row_dict["distance"])

            matches.append(
                LLMRouteMatch(
                    tier=tier_name,
                    model=tier.model if tier else None,
                    distance=distance,
                    confidence=1 - (distance / 2),
                    metadata=tier.metadata if tier else {},
                )
            )

        return matches

    def _apply_cost_optimization(
        self, matches: List[LLMRouteMatch]
    ) -> List[LLMRouteMatch]:
        """Re-rank matches considering cost."""
        if not matches or not self.routing_config.cost_optimization:
            return matches

        cost_weight = self.routing_config.cost_weight
        ranked = []

        for match in matches:
            cost = match.metadata.get("cost_per_1k_input", 0)
            adjusted_distance = match.distance + (cost * cost_weight)
            ranked.append((match, adjusted_distance))

        ranked.sort(key=lambda x: x[1])
        return [m for m, _ in ranked]

    async def route(
        self,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        aggregation_method: Optional[DistanceAggregationMethod] = None,
    ) -> LLMRouteMatch:
        """Route a query to the best matching tier (async).

        Args:
            query: Text query to route.
            vector: Pre-computed embedding vector.
            aggregation_method: Override default aggregation method.

        Returns:
            LLMRouteMatch with tier, model, and routing metadata.
        """
        if vector is None:
            if query is None:
                raise ValueError("Must provide query or vector")
            vector = await self.vectorizer.aembed(query)

        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        matches = await self._get_tier_matches(
            vector, aggregation_method, max_k=len(self.tiers)
        )
        matches = self._apply_cost_optimization(matches)

        if not matches:
            if self.default_tier:
                tier = self.get_tier(self.default_tier)
                if tier:
                    return LLMRouteMatch(
                        tier=tier.name,
                        model=tier.model,
                        metadata=tier.metadata,
                    )
            return LLMRouteMatch()

        top_match = matches[0]
        top_match.alternatives = [(m.tier, m.distance) for m in matches[1:]]
        return top_match

    async def route_many(
        self,
        query: Optional[str] = None,
        vector: Optional[List[float]] = None,
        max_k: Optional[int] = None,
        aggregation_method: Optional[DistanceAggregationMethod] = None,
    ) -> List[LLMRouteMatch]:
        """Route a query and return multiple tier matches (async).

        Args:
            query: Text query to route.
            vector: Pre-computed embedding vector.
            max_k: Maximum number of matches to return.
            aggregation_method: Override default aggregation method.

        Returns:
            List of LLMRouteMatch objects ordered by distance.
        """
        if vector is None:
            if query is None:
                raise ValueError("Must provide query or vector")
            vector = await self.vectorizer.aembed(query)

        max_k = max_k or self.routing_config.max_k
        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        matches = await self._get_tier_matches(vector, aggregation_method, max_k)
        return self._apply_cost_optimization(matches)

    # Serialization methods

    def to_dict(self) -> Dict[str, Any]:
        """Serialize router to dictionary."""
        return {
            "name": self.name,
            "tiers": [model_to_dict(tier) for tier in self.tiers],
            "vectorizer": {
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            "routing_config": model_to_dict(self.routing_config),
        }

    @classmethod
    async def from_dict(
        cls,
        data: Dict[str, Any],
        **kwargs,
    ) -> "AsyncLLMRouter":
        """Create router from dictionary (async)."""
        from redisvl.utils.vectorize import vectorizer_from_dict

        try:
            name = data["name"]
            tiers_data = data["tiers"]
            vectorizer_data = data["vectorizer"]
            routing_config_data = data.get("routing_config", {})
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")

        vectorizer = vectorizer_from_dict(vectorizer_data)
        if not vectorizer:
            raise ValueError(f"Could not load vectorizer: {vectorizer_data}")

        tiers = [ModelTier(**t) for t in tiers_data]
        routing_config = RoutingConfig(**routing_config_data)

        return await cls.create(
            name=name,
            tiers=tiers,
            vectorizer=vectorizer,
            routing_config=routing_config,
            **kwargs,
        )

    def to_yaml(self, file_path: str, overwrite: bool = True):
        """Save router to YAML file."""
        fp = Path(file_path).resolve()
        if fp.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists")

        with open(fp, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    async def from_yaml(cls, file_path: str, **kwargs) -> "AsyncLLMRouter":
        """Load router from YAML file (async)."""
        fp = Path(file_path).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        with open(fp) as f:
            data = yaml.safe_load(f)

        return await cls.from_dict(data, **kwargs)

    async def export_with_embeddings(self, file_path: str):
        """Export router with pre-computed embeddings (async).

        Args:
            file_path: Path to JSON file.
        """
        fp = Path(file_path).resolve()

        pretrained_tiers = []
        for tier in self.tiers:
            vectors = await self.vectorizer.aembed_many(tier.references)

            references = [
                PretrainedReference(text=ref, vector=vec)
                for ref, vec in zip(tier.references, vectors)
            ]

            pretrained_tiers.append(
                PretrainedTier(
                    name=tier.name,
                    model=tier.model,
                    references=references,
                    metadata=tier.metadata,
                    distance_threshold=tier.distance_threshold,
                )
            )

        config = PretrainedRouterConfig(
            name=self.name,
            vectorizer={
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            tiers=pretrained_tiers,
            routing_config=model_to_dict(self.routing_config),
        )

        with open(fp, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

    @classmethod
    async def from_pretrained(
        cls,
        config_name_or_path: str,
        redis_client: Optional[AsyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        **kwargs,
    ) -> "AsyncLLMRouter":
        """Load router from pretrained config with embeddings (async).

        Accepts either a file path or a built-in config name (e.g., "default").

        Args:
            config_name_or_path: Path to pretrained JSON file, or name of a
                built-in config (e.g., "default").
            redis_client: Async Redis client.
            redis_url: Redis URL.

        Returns:
            AsyncLLMRouter loaded without re-embedding.
        """
        import numpy as np

        from redisvl.utils.vectorize import vectorizer_from_dict

        fp = Path(config_name_or_path)
        if not fp.exists():
            from redisvl.extensions.llm_router.pretrained import get_pretrained_path

            fp = get_pretrained_path(config_name_or_path)

        with open(fp) as f:
            data = json.load(f)

        config = PretrainedRouterConfig(**data)

        vectorizer = vectorizer_from_dict(config.vectorizer)
        if not vectorizer:
            raise ValueError(f"Could not load vectorizer: {config.vectorizer}")

        # Prefer provided client over URL
        if redis_client:
            await RedisConnectionFactory.validate_async_redis(redis_client)
        elif redis_url:
            redis_client = await RedisConnectionFactory._get_aredis_connection(
                redis_url=redis_url, **kwargs
            )

        if redis_client is None:
            raise ValueError("Could not establish Redis connection")

        schema = SemanticRouterIndexSchema.from_params(
            config.name, vectorizer.dims, vectorizer.dtype  # type: ignore
        )

        index = AsyncSearchIndex(
            schema=schema,
            redis_client=redis_client,
        )
        await index.create(overwrite=True, drop=False)

        tiers = []
        all_references = []
        all_keys = []

        for pt in config.tiers:
            tier = ModelTier(
                name=pt.name,
                model=pt.model,
                references=[r.text for r in pt.references],
                metadata=pt.metadata,
                distance_threshold=pt.distance_threshold,
            )
            tiers.append(tier)

            for ref in pt.references:
                vector_buffer = np.array(ref.vector, dtype=np.float32).tobytes()
                reference_hash = hashify(ref.text)

                sep = index.key_separator
                prefix = (
                    index.prefix.rstrip(sep) if sep and index.prefix else index.prefix
                )
                if prefix:
                    key = f"{prefix}{sep}{pt.name}{sep}{reference_hash}"
                else:
                    key = f"{pt.name}{sep}{reference_hash}"

                all_references.append(
                    {
                        "reference_id": reference_hash,
                        "route_name": pt.name,
                        "reference": ref.text,
                        "vector": vector_buffer,
                    }
                )
                all_keys.append(key)

        await index.load(all_references, keys=all_keys)

        routing_config = RoutingConfig(**config.routing_config)

        router = cls.model_construct(
            name=config.name,
            tiers=tiers,
            vectorizer=vectorizer,
            routing_config=routing_config,
        )
        object.__setattr__(router, "_index", index)

        await redis_client.json().set(  # type: ignore
            f"{config.name}:router_config", ".", router.to_dict()
        )

        return router

    async def _update_router_config(self):
        """Update router config in Redis."""
        client = self._index.client
        await client.json().set(f"{self.name}:router_config", ".", self.to_dict())  # type: ignore

    async def delete(self):
        """Delete the router index."""
        client = self._index.client
        await client.delete(f"{self.name}:router_config")  # type: ignore
        await self._index.delete(drop=True)

    async def clear(self):
        """Clear all tier references."""
        await self._index.clear()
        self.tiers = []
