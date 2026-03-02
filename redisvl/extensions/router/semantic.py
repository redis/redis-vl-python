import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, Union

import redis.commands.search.reducers as reducers
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from redis.commands.search.aggregation import AggregateRequest, AggregateResult, Reducer
from redis.exceptions import ResponseError

from redisvl.extensions.constants import ROUTE_VECTOR_FIELD_NAME
from redisvl.extensions.router.schema import (
    DistanceAggregationMethod,
    Route,
    RouteMatch,
    RoutingConfig,
    SemanticRouterIndexSchema,
)
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import FilterQuery, VectorRangeQuery
from redisvl.query.filter import Tag
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.redis.utils import convert_bytes, hashify, make_dict
from redisvl.types import AsyncRedisClient, SyncRedisClient
from redisvl.utils.log import get_logger
from redisvl.utils.utils import deprecated_argument, model_to_dict, scan_by_pattern
from redisvl.utils.vectorize.base import BaseVectorizer
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

logger = get_logger(__name__)


class SemanticRouter(BaseModel):
    """Semantic Router for managing and querying route vectors."""

    name: str
    """The name of the semantic router."""
    routes: List[Route]
    """List of Route objects."""
    vectorizer: BaseVectorizer = Field(default_factory=HFTextVectorizer)
    """The vectorizer used to embed route references."""
    routing_config: RoutingConfig = Field(default_factory=RoutingConfig)
    """Configuration for routing behavior."""

    _index: SearchIndex = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @deprecated_argument("dtype", "vectorizer")
    def __init__(
        self,
        name: str,
        routes: List[Route],
        vectorizer: Optional[BaseVectorizer] = None,
        routing_config: Optional[RoutingConfig] = None,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        connection_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the SemanticRouter.

        Args:
            name (str): The name of the semantic router.
            routes (List[Route]): List of Route objects.
            vectorizer (BaseVectorizer, optional): The vectorizer used to embed route references. Defaults to default HFTextVectorizer.
            routing_config (RoutingConfig, optional): Configuration for routing behavior. Defaults to the default RoutingConfig.
            redis_client (Optional[SyncRedisClient], optional): Redis client for connection. Defaults to None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            overwrite (bool, optional): Whether to overwrite existing index. Defaults to False.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.
        """
        dtype = kwargs.pop("dtype", None)

        # Validate a provided vectorizer or set the default
        if vectorizer:
            if not isinstance(vectorizer, BaseVectorizer):
                raise TypeError("Must provide a valid redisvl.vectorizer class.")
            if dtype and vectorizer.dtype != dtype:
                raise ValueError(
                    f"Provided dtype {dtype} does not match vectorizer dtype {vectorizer.dtype}"
                )
        else:
            vectorizer_kwargs = kwargs

            if dtype:
                vectorizer_kwargs.update(**{"dtype": dtype})

            vectorizer = HFTextVectorizer(
                model="sentence-transformers/all-mpnet-base-v2",
                **vectorizer_kwargs,
            )

        if routing_config is None:
            routing_config = RoutingConfig()

        if connection_kwargs is None:
            connection_kwargs = {}

        super().__init__(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
            redis_url=redis_url,
            redis_client=redis_client,
        )

        self._initialize_index(redis_client, redis_url, overwrite, **connection_kwargs)

        self._index.client.json().set(f"{self.name}:route_config", f".", self.to_dict())  # type: ignore

    @classmethod
    def from_existing(
        cls,
        name: str,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        **kwargs,
    ) -> "SemanticRouter":
        """Return SemanticRouter instance from existing index."""
        if redis_url:
            redis_client = RedisConnectionFactory.get_redis_connection(
                redis_url=redis_url,
                **kwargs,
            )
        elif redis_client:
            # Just validate client type and set lib name
            RedisConnectionFactory.validate_sync_redis(redis_client)
        if redis_client is None:
            raise ValueError(
                "Creating Redis client failed. Please check the redis_url and connection_kwargs."
            )

        router_dict = redis_client.json().get(f"{name}:route_config")
        if not isinstance(router_dict, dict):
            raise ValueError(
                f"No valid router config found for {name}. Received: {router_dict!r}"
            )
        return cls.from_dict(
            router_dict, redis_url=redis_url, redis_client=redis_client
        )

    @deprecated_argument("dtype")
    def _initialize_index(
        self,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        dtype: str = "float32",
        **connection_kwargs,
    ):
        """Initialize the search index and handle Redis connection."""

        schema = SemanticRouterIndexSchema.from_params(
            self.name, self.vectorizer.dims, self.vectorizer.dtype  # type: ignore
        )

        self._index = SearchIndex(
            schema=schema,
            redis_client=redis_client,
            redis_url=redis_url,
            **connection_kwargs,
        )

        # Check for existing router index
        existed = self._index.exists()
        if not overwrite and existed:
            existing_index = SearchIndex.from_existing(
                self.name, redis_client=self._index.client
            )
            if existing_index.schema.to_dict() != self._index.schema.to_dict():
                raise ValueError(
                    f"Existing index {self.name} schema does not match the user provided schema for the semantic router. "
                    "If you wish to overwrite the index schema, set overwrite=True during initialization."
                )
        self._index.create(overwrite=overwrite, drop=False)

        if not existed or overwrite:
            # write the routes to Redis
            self._add_routes(self.routes)

    @property
    def route_names(self) -> List[str]:
        """Get the list of route names.

        Returns:
            List[str]: List of route names.
        """
        return [route.name for route in self.routes]

    @property
    def route_thresholds(self) -> Dict[str, Optional[float]]:
        """Get the distance thresholds for each route.

        Returns:
            Dict[str, float]: Dictionary of route names and their distance thresholds.
        """
        return {route.name: route.distance_threshold for route in self.routes}

    def update_routing_config(self, routing_config: RoutingConfig):
        """Update the routing configuration.

        Args:
            routing_config (RoutingConfig): The new routing configuration.
        """
        self.routing_config = routing_config

    def update_route_thresholds(self, route_thresholds: Dict[str, Optional[float]]):
        """Update the distance thresholds for each route.

        Args:
            route_thresholds (Dict[str, float]): Dictionary of route names and their distance thresholds.
        """
        for route in self.routes:
            if route.name in route_thresholds:
                route.distance_threshold = route_thresholds[route.name]  # type: ignore

    @staticmethod
    def _route_ref_key(index: SearchIndex, route_name: str, reference_hash: str) -> str:
        """Generate the route reference key using the index's key_separator."""
        sep = index.key_separator
        # Normalize prefix to avoid double separators
        prefix = index.prefix.rstrip(sep) if sep and index.prefix else index.prefix
        if prefix:
            return f"{prefix}{sep}{route_name}{sep}{reference_hash}"
        else:
            return f"{route_name}{sep}{reference_hash}"

    @staticmethod
    def _route_pattern(index: SearchIndex, route_name: str) -> str:
        """Generate a search pattern for route references."""
        sep = index.key_separator
        # Normalize prefix to avoid double separators
        prefix = index.prefix.rstrip(sep) if sep and index.prefix else index.prefix
        if prefix:
            return f"{prefix}{sep}{route_name}{sep}*"
        else:
            return f"{route_name}{sep}*"

    def _add_routes(self, routes: List[Route]):
        """Add routes to the router and index.

        Args:
            routes (List[Route]): List of routes to be added.
        """
        route_references: List[Dict[str, Any]] = []
        keys: List[str] = []

        for route in routes:
            # embed route references as a single batch
            reference_vectors = self.vectorizer.embed_many(
                [reference for reference in route.references], as_buffer=True
            )
            # set route references
            for i, reference in enumerate(route.references):
                reference_hash = hashify(reference)
                route_references.append(
                    {
                        "reference_id": reference_hash,
                        "route_name": route.name,
                        "reference": reference,
                        "vector": reference_vectors[i],
                    }
                )
                keys.append(
                    self._route_ref_key(self._index, route.name, reference_hash)
                )

            # set route if does not yet exist client side
            if not self.get(route.name):
                self.routes.append(route)

        self._index.load(route_references, keys=keys)

    def get(self, route_name: str) -> Optional[Route]:
        """Get a route by its name.

        Args:
            route_name (str): Name of the route.

        Returns:
            Optional[Route]: The selected Route object or None if not found.
        """
        return next((route for route in self.routes if route.name == route_name), None)

    def _process_route(self, result: Dict[str, Any]) -> RouteMatch:
        """Process resulting route objects and metadata."""
        route_dict = make_dict(convert_bytes(result))
        route_name = route_dict["route_name"]
        distance = float(route_dict["distance"])

        # Get route to extract model and metadata if available
        route = self.get(route_name)

        return RouteMatch(
            name=route_name,
            distance=distance,
            model=route.model if route and route.model else None,
            confidence=(1 - distance / 2) if route and route.model else None,
            metadata=route.metadata if route else {},
        )

    def _apply_cost_optimization(self, matches: List[RouteMatch]) -> List[RouteMatch]:
        """Re-rank matches considering cost when cost_optimization is enabled.

        Adjusts match distances by adding a cost penalty based on the
        cost_per_1k_input metadata field and cost_weight configuration.

        Args:
            matches: List of route matches to re-rank.

        Returns:
            Re-ranked list of matches (or original if cost optimization disabled).
        """
        if not matches or not self.routing_config.cost_optimization:
            return matches

        cost_weight = self.routing_config.cost_weight
        ranked = []

        for match in matches:
            cost = match.metadata.get("cost_per_1k_input", 0)
            # Add cost penalty to distance (normalized)
            adjusted_distance = match.distance + (cost * cost_weight)  # type: ignore
            ranked.append((match, adjusted_distance))

        # Re-sort by adjusted distance
        ranked.sort(key=lambda x: x[1])

        # Update alternatives with original matches
        for i, (match, _) in enumerate(ranked):
            match.alternatives = [
                (m.name, m.distance) for j, (m, _) in enumerate(ranked) if j != i
            ]

        return [m for m, _ in ranked]

    def _distance_threshold_filter(self) -> str:
        """Apply distance threshold on a route by route basis."""
        filter = ""
        for i, route in enumerate(self.routes):
            filter_str = f"(@route_name == '{route.name}' && @distance < {route.distance_threshold})"
            if i > 0:
                filter += " || "
            filter += filter_str

        return filter

    def _build_aggregate_request(
        self,
        vector_range_query: VectorRangeQuery,
        aggregation_method: DistanceAggregationMethod,
        max_k: int,
    ) -> AggregateRequest:
        """Build the Redis aggregation request."""
        aggregation_func: Type[Reducer]

        if aggregation_method == DistanceAggregationMethod.min:
            aggregation_func = reducers.min
        elif aggregation_method == DistanceAggregationMethod.sum:
            aggregation_func = reducers.sum
        else:
            aggregation_func = reducers.avg

        aggregate_query = str(vector_range_query).split(" RETURN")[0]
        aggregate_request = (
            AggregateRequest(aggregate_query)
            .group_by(
                "@route_name",  # type: ignore
                aggregation_func("vector_distance").alias("distance"),
            )
            .sort_by("@distance", max=max_k)  # type: ignore
            .dialect(2)
        )

        filter = self._distance_threshold_filter()

        aggregate_request.filter(filter)

        return aggregate_request

    def _get_route_matches(
        self,
        vector: List[float],
        aggregation_method: DistanceAggregationMethod,
        max_k: int = 1,
    ) -> List[RouteMatch]:
        """Get route response from vector db"""
        if not self.routes:
            return []

        # what's interesting about this is that we only provide one distance_threshold for a range query not multiple
        # therefore you might take the max_threshold and further refine from there.
        distance_threshold = max(route.distance_threshold for route in self.routes)

        vector_range_query = VectorRangeQuery(
            vector=vector,
            vector_field_name=ROUTE_VECTOR_FIELD_NAME,
            distance_threshold=float(distance_threshold),
            return_fields=["route_name"],
        )

        aggregate_request = self._build_aggregate_request(
            vector_range_query, aggregation_method, max_k=max_k
        )

        try:
            aggregation_result: AggregateResult = self._index.aggregate(
                aggregate_request, vector_range_query.params
            )
        except ResponseError as e:
            if "VSS is not yet supported on FT.AGGREGATE" in str(e):
                raise RuntimeError(
                    "Semantic routing is only available on Redis version 7.x.x or greater"
                )
            raise e

        # process aggregation results into route matches
        return [
            self._process_route(route_match) for route_match in aggregation_result.rows
        ]

    def _classify_route(
        self,
        vector: List[float],
        aggregation_method: DistanceAggregationMethod,
    ) -> RouteMatch:
        """Classify to a single route using a vector."""

        # Get all potential route matches
        route_matches = self._get_route_matches(
            vector, aggregation_method, max_k=len(self.routes)
        )

        if not route_matches:
            # Return default route if configured
            if self.routing_config.default_route:
                route = self.get(self.routing_config.default_route)
                if route:
                    return RouteMatch(
                        name=route.name,
                        model=route.model,
                        metadata=route.metadata,
                    )
            return RouteMatch()

        # Apply cost optimization if enabled
        route_matches = self._apply_cost_optimization(route_matches)

        # Take top route after optimization
        top_route_match = route_matches[0]

        if top_route_match.name is not None:
            return top_route_match
        else:
            raise ValueError(
                f"{top_route_match.name} not a supported route for the {self.name} semantic router."
            )

    def _classify_multi_route(
        self,
        vector: List[float],
        max_k: int,
        aggregation_method: DistanceAggregationMethod,
    ) -> List[RouteMatch]:
        """Classify to multiple routes, up to max_k (int), using a vector."""

        route_matches = self._get_route_matches(vector, aggregation_method, max_k=max_k)

        # Apply cost optimization if enabled
        route_matches = self._apply_cost_optimization(route_matches)

        # Process route matches
        top_route_matches: List[RouteMatch] = []
        if route_matches:
            for route_match in route_matches:
                if route_match.name is not None:
                    top_route_matches.append(route_match)
                else:
                    raise ValueError(
                        f"{route_match.name} not a supported route for the {self.name} semantic router."
                    )

        return top_route_matches

    @deprecated_argument("distance_threshold")
    def __call__(
        self,
        statement: Optional[str] = None,
        vector: Optional[List[float]] = None,
        aggregation_method: Optional[DistanceAggregationMethod] = None,
        distance_threshold: Optional[float] = None,
    ) -> RouteMatch:
        """Query the semantic router with a given statement or vector.

        Args:
            statement (Optional[str]): The input statement to be queried.
            vector (Optional[List[float]]): The input vector to be queried.
            distance_threshold (Optional[float]): The threshold for semantic distance.
            aggregation_method (Optional[DistanceAggregationMethod]): The aggregation method used for vector distances.

        Returns:
            RouteMatch: The matching route.
        """
        if not vector:
            if not statement:
                raise ValueError("Must provide a vector or statement to the router")
            vector = self.vectorizer.embed(statement)  # type: ignore

        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        # perform route classification
        top_route_match = self._classify_route(vector, aggregation_method)  # type: ignore
        return top_route_match

    @deprecated_argument("distance_threshold")
    def route_many(
        self,
        statement: Optional[str] = None,
        vector: Optional[List[float]] = None,
        max_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        aggregation_method: Optional[DistanceAggregationMethod] = None,
    ) -> List[RouteMatch]:
        """Query the semantic router with a given statement or vector for multiple matches.

        Args:
            statement (Optional[str]): The input statement to be queried.
            vector (Optional[List[float]]): The input vector to be queried.
            max_k (Optional[int]): The maximum number of top matches to return.
            distance_threshold (Optional[float]): The threshold for semantic distance.
            aggregation_method (Optional[DistanceAggregationMethod]): The aggregation method used for vector distances.

        Returns:
            List[RouteMatch]: The matching routes and their details.
        """
        if not vector:
            if not statement:
                raise ValueError("Must provide a vector or statement to the router")
            vector = self.vectorizer.embed(statement)  # type: ignore

        max_k = max_k or self.routing_config.max_k
        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        # classify routes
        top_route_matches = self._classify_multi_route(
            vector, max_k, aggregation_method  # type: ignore
        )

        return top_route_matches

    def remove_route(self, route_name: str) -> None:
        """Remove a route and all references from the semantic router.

        Args:
            route_name (str): Name of the route to remove.
        """
        route = self.get(route_name)
        if route is None:
            logger.warning(f"Route {route_name} is not found in the SemanticRouter")
        else:
            self._index.drop_keys(
                [
                    self._route_ref_key(self._index, route.name, hashify(reference))
                    for reference in route.references
                ]
            )
            self.routes = [route for route in self.routes if route.name != route_name]

    def delete(self) -> None:
        """Delete the semantic router index and associated config."""
        self._index.delete(drop=True)
        # Clean up router config key
        self._index.client.delete(f"{self.name}:route_config")  # type: ignore

    def clear(self) -> None:
        """Flush all routes from the semantic router index."""
        self._index.clear()
        self.routes = []

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        **kwargs,
    ) -> "SemanticRouter":
        """Create a SemanticRouter from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary containing the semantic router data.

        Returns:
            SemanticRouter: The semantic router instance.

        Raises:
            ValueError: If required data is missing or invalid.

        .. code-block:: python

            from redisvl.extensions.router import SemanticRouter
            router_data = {
                "name": "example_router",
                "routes": [{"name": "route1", "references": ["ref1"], "distance_threshold": 0.5}],
                "vectorizer": {"type": "openai", "model": "text-embedding-ada-002"},
            }
            router = SemanticRouter.from_dict(router_data)
        """
        from redisvl.utils.vectorize import vectorizer_from_dict

        try:
            name = data["name"]
            routes_data = data["routes"]
            vectorizer_data = data["vectorizer"]
            routing_config_data = data["routing_config"]
        except KeyError as e:
            raise ValueError(f"Unable to load semantic router from dict: {str(e)}")

        try:
            vectorizer = vectorizer_from_dict(vectorizer_data)
        except Exception as e:
            raise ValueError(f"Unable to load vectorizer: {str(e)}")

        if not vectorizer:
            raise ValueError(f"Unable to load vectorizer: {vectorizer_data}")

        routes = [Route(**route) for route in routes_data]
        routing_config = RoutingConfig(**routing_config_data)

        return cls(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the SemanticRouter instance to a dictionary.

        Returns:
            Dict[str, Any]: The dictionary representation of the SemanticRouter.

        .. code-block:: python

            from redisvl.extensions.router import SemanticRouter
            router = SemanticRouter(name="example_router", routes=[], redis_url="redis://localhost:6379")
            router_dict = router.to_dict()
        """
        return {
            "name": self.name,
            "routes": [model_to_dict(route) for route in self.routes],
            "vectorizer": {
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            "routing_config": model_to_dict(self.routing_config),
        }

    @classmethod
    def from_yaml(
        cls,
        file_path: str,
        **kwargs,
    ) -> "SemanticRouter":
        """Create a SemanticRouter from a YAML file.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            SemanticRouter: The semantic router instance.

        Raises:
            ValueError: If the file path is invalid.
            FileNotFoundError: If the file does not exist.

        .. code-block:: python

            from redisvl.extensions.router import SemanticRouter
            router = SemanticRouter.from_yaml("router.yaml", redis_url="redis://localhost:6379")
        """
        try:
            fp = Path(file_path).resolve()
        except OSError as e:
            raise ValueError(f"Invalid file path: {file_path}") from e

        if not fp.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        with open(fp, "r") as f:
            yaml_data = yaml.safe_load(f)
            return cls.from_dict(
                yaml_data,
                **kwargs,
            )

    @classmethod
    def from_pretrained(
        cls,
        config_name_or_path: str,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = True,
        **kwargs,
    ) -> "SemanticRouter":
        """Load router from pretrained config with embeddings.

        This skips the embedding step by using pre-computed vectors.
        Accepts either a file path or a built-in config name (e.g., "default").

        Args:
            config_name_or_path: Path to pretrained JSON file, or name of a
                built-in config (e.g., "default").
            redis_client: Redis client.
            redis_url: Redis URL.
            overwrite: Whether to overwrite an existing index with the same
                name. Defaults to True.
            **kwargs: Additional connection arguments.

        Returns:
            SemanticRouter loaded without re-embedding.

        .. code-block:: python

            from redisvl.extensions.router import SemanticRouter

            # Load built-in pretrained router
            router = SemanticRouter.from_pretrained("default")

            # Or load from custom file
            router = SemanticRouter.from_pretrained("my_router.json")
        """
        import json

        import numpy as np

        from redisvl.extensions.router.pretrained import get_pretrained_path
        from redisvl.extensions.router.schema import PretrainedRouterConfig
        from redisvl.utils.vectorize import vectorizer_from_dict

        fp = Path(config_name_or_path)
        if not fp.exists():
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
        index.create(overwrite=overwrite, drop=False)

        # Load pre-computed embeddings directly
        routes = []
        all_references = []
        all_keys = []

        for pr in config.routes:
            route = Route(
                name=pr.name,
                references=[r.text for r in pr.references],
                metadata=pr.metadata,
                distance_threshold=pr.distance_threshold,
                model=pr.model,
            )
            routes.append(route)

            # Use pre-computed vectors (convert to buffer)
            for ref in pr.references:
                vector_buffer = np.array(ref.vector, dtype=np.float32).tobytes()
                reference_hash = hashify(ref.text)

                sep = index.key_separator
                prefix = (
                    index.prefix.rstrip(sep) if sep and index.prefix else index.prefix
                )
                if prefix:
                    key = f"{prefix}{sep}{pr.name}{sep}{reference_hash}"
                else:
                    key = f"{pr.name}{sep}{reference_hash}"

                all_references.append(
                    {
                        "reference_id": reference_hash,
                        "route_name": pr.name,
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
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
        )
        # Set private attribute directly
        object.__setattr__(router, "_index", index)

        # Store config
        redis_client.json().set(f"{config.name}:route_config", ".", router.to_dict())

        return router

    def export_with_embeddings(self, file_path: str):
        """Export router with pre-computed embeddings.

        This allows loading the router without re-embedding references.

        Args:
            file_path: Path to JSON file.

        .. code-block:: python

            from redisvl.extensions.router import SemanticRouter

            router = SemanticRouter(name="my_router", routes=routes)
            router.export_with_embeddings("my_router.json")

            # Later, load without re-embedding
            router = SemanticRouter.from_pretrained("my_router.json")
        """
        import json

        from redisvl.extensions.router.schema import (
            PretrainedReference,
            PretrainedRoute,
            PretrainedRouterConfig,
        )

        fp = Path(file_path).resolve()

        pretrained_routes = []
        for route in self.routes:
            # Get embeddings for all references
            vectors = self.vectorizer.embed_many(route.references)

            references = [
                PretrainedReference(text=ref, vector=vec)
                for ref, vec in zip(route.references, vectors)
            ]

            pretrained_routes.append(
                PretrainedRoute(
                    name=route.name,
                    references=references,
                    metadata=route.metadata,
                    distance_threshold=route.distance_threshold,
                    model=route.model,
                )
            )

        config = PretrainedRouterConfig(
            name=self.name,
            vectorizer={
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            routes=pretrained_routes,
            routing_config=model_to_dict(self.routing_config),
        )

        with open(fp, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

    def to_yaml(self, file_path: str, overwrite: bool = True) -> None:
        """Write the semantic router to a YAML file.

        Args:
            file_path (str): The path to the YAML file.
            overwrite (bool): Whether to overwrite the file if it already exists.

        Raises:
            FileExistsError: If the file already exists and overwrite is False.

        .. code-block:: python

            from redisvl.extensions.router import SemanticRouter
            router = SemanticRouter(
                name="example_router",
                routes=[],
                redis_url="redis://localhost:6379"
            )
            router.to_yaml("router.yaml")
        """
        fp = Path(file_path).resolve()
        if fp.exists() and not overwrite:
            raise FileExistsError(f"Schema file {file_path} already exists.")

        with open(fp, "w") as f:
            yaml_data = self.to_dict()
            yaml.dump(yaml_data, f, sort_keys=False)

    # reference methods
    def add_route_references(
        self,
        route_name: str,
        references: Union[str, List[str]],
    ) -> List[str]:
        """Add a reference(s) to an existing route.

        Args:
            router_name (str): The name of the router.
            references (Union[str, List[str]]): The reference or list of references to add.

        Returns:
            List[str]: The list of added references keys.
        """

        if isinstance(references, str):
            references = [references]

        route_references: List[Dict[str, Any]] = []
        keys: List[str] = []

        # embed route references as a single batch
        reference_vectors = self.vectorizer.embed_many(references, as_buffer=True)

        # set route references
        for i, reference in enumerate(references):
            reference_hash = hashify(reference)

            route_references.append(
                {
                    "reference_id": reference_hash,
                    "route_name": route_name,
                    "reference": reference,
                    "vector": reference_vectors[i],
                }
            )
            keys.append(self._route_ref_key(self._index, route_name, reference_hash))

        keys = self._index.load(route_references, keys=keys)

        route = self.get(route_name)
        if not route:
            raise ValueError(f"Route {route_name} not found in the SemanticRouter")
        route.references.extend(references)
        self._update_router_state()
        return keys

    @staticmethod
    def _make_filter_queries(ids: List[str]) -> List[FilterQuery]:
        """Create a filter query for the given ids."""

        queries = []

        for id in ids:
            fe = Tag("reference_id") == id
            fq = FilterQuery(
                return_fields=["reference_id", "route_name", "reference"],
                filter_expression=fe,
            )
            queries.append(fq)

        return queries

    def get_route_references(
        self,
        route_name: str = "",
        reference_ids: List[str] = [],
        keys: List[str] = [],
    ) -> List[Dict[str, Any]]:
        """Get references for an existing route route.

        Args:
            router_name (str): The name of the router.
            references (Union[str, List[str]]): The reference or list of references to add.

        Returns:
            List[Dict[str, Any]]]: Reference objects stored
        """

        if reference_ids:
            queries = self._make_filter_queries(reference_ids)
        elif route_name:
            if not keys:
                pattern = self._route_pattern(self._index, route_name)
                keys = scan_by_pattern(self._index.client, pattern)  # type: ignore

            sep = self._index.key_separator
            queries = self._make_filter_queries(
                [key.split(sep)[-1] for key in convert_bytes(keys)]
            )
        else:
            raise ValueError(
                "Must provide a route name, reference ids, or keys to get references"
            )

        res = self._index.batch_query(queries)

        return [r[0] for r in res if len(r) > 0]

    def delete_route_references(
        self,
        route_name: str = "",
        reference_ids: List[str] = [],
        keys: List[str] = [],
    ) -> int:
        """Get references for an existing semantic router route.

        Args:
            router_name Optional(str): The name of the router.
            reference_ids Optional(List[str]]): The reference or list of references to delete.
            keys Optional(List[str]]): List of fully qualified keys (prefix:router:reference_id) to delete.

        Returns:
            int: Number of objects deleted
        """

        if reference_ids and not keys:
            queries = self._make_filter_queries(reference_ids)
            res = self._index.batch_query(queries)
            keys = [r[0]["id"] for r in res if len(r) > 0]
        elif not keys:
            pattern = self._route_pattern(self._index, route_name)
            keys = scan_by_pattern(self._index.client, pattern)  # type: ignore

        if not keys:
            raise ValueError(f"No references found for route {route_name}")

        to_be_deleted = []
        for key in keys:
            route_name = key.split(":")[-2]
            to_be_deleted.append(
                (route_name, convert_bytes(self._index.client.hgetall(key)))  # type: ignore
            )

        deleted = self._index.drop_keys(keys)

        for route_name, delete in to_be_deleted:
            route = self.get(route_name)
            if not route:
                raise ValueError(f"Route {route_name} not found in the SemanticRouter")
            route.references.remove(delete["reference"])

        self._update_router_state()

        return deleted

    def _update_router_state(self) -> None:
        """Update the router configuration in Redis."""
        self._index.client.json().set(f"{self.name}:route_config", f".", self.to_dict())  # type: ignore


class AsyncSemanticRouter(BaseModel):
    """Async Semantic Router for managing and querying route vectors.

    Provides the same functionality as :class:`SemanticRouter` but uses async I/O.
    Since ``__init__`` cannot be async, use the :meth:`create` classmethod
    factory to instantiate.
    """

    name: str
    """The name of the semantic router."""
    routes: List[Route]
    """List of Route objects."""
    vectorizer: BaseVectorizer = Field(default_factory=HFTextVectorizer)
    """The vectorizer used to embed route references."""
    routing_config: RoutingConfig = Field(default_factory=RoutingConfig)
    """Configuration for routing behavior."""

    _index: AsyncSearchIndex = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    async def create(
        cls,
        name: str,
        routes: List[Route],
        vectorizer: Optional[BaseVectorizer] = None,
        routing_config: Optional[RoutingConfig] = None,
        redis_client: Optional[AsyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        connection_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "AsyncSemanticRouter":
        """Create an AsyncSemanticRouter instance (async factory).

        Args:
            name (str): The name of the semantic router.
            routes (List[Route]): List of Route objects.
            vectorizer (BaseVectorizer, optional): The vectorizer used to embed route references. Defaults to default HFTextVectorizer.
            routing_config (RoutingConfig, optional): Configuration for routing behavior. Defaults to the default RoutingConfig.
            redis_client (Optional[AsyncRedisClient], optional): Async Redis client for connection. Defaults to None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            overwrite (bool, optional): Whether to overwrite existing index. Defaults to False.
            connection_kwargs (Optional[Dict[str, Any]]): The connection arguments
                for the redis client. Defaults to None.

        Returns:
            AsyncSemanticRouter: The async semantic router instance.
        """
        if connection_kwargs is None:
            connection_kwargs = {}

        # Validate a provided vectorizer or set the default
        if vectorizer:
            if not isinstance(vectorizer, BaseVectorizer):
                raise TypeError("Must provide a valid redisvl.vectorizer class.")
        else:
            vectorizer = HFTextVectorizer(
                model="sentence-transformers/all-mpnet-base-v2",
            )

        if routing_config is None:
            routing_config = RoutingConfig()

        router = cls.model_construct(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
        )

        await router._initialize_index(
            redis_client, redis_url, overwrite, **connection_kwargs
        )

        client = router._index.client
        await client.json().set(f"{name}:route_config", ".", router.to_dict())  # type: ignore
        return router

    @classmethod
    async def from_existing(
        cls,
        name: str,
        redis_client: Optional[AsyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        **kwargs,
    ) -> "AsyncSemanticRouter":
        """Return AsyncSemanticRouter instance from existing index.

        Args:
            name (str): The name of the semantic router.
            redis_client (Optional[AsyncRedisClient]): Async Redis client. Defaults to None.
            redis_url (str): Redis connection URL. Defaults to "redis://localhost:6379".

        Returns:
            AsyncSemanticRouter: The async semantic router instance.
        """
        if redis_client:
            await RedisConnectionFactory.validate_async_redis(redis_client)
        elif redis_url:
            redis_client = await RedisConnectionFactory._get_aredis_connection(
                redis_url=redis_url,
                **kwargs,
            )

        if redis_client is None:
            raise ValueError(
                "Creating Redis client failed. Please check the redis_url and connection_kwargs."
            )

        router_dict = await redis_client.json().get(f"{name}:route_config")  # type: ignore
        if not isinstance(router_dict, dict):
            raise ValueError(
                f"No valid router config found for {name}. Received: {router_dict!r}"
            )
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
        """Initialize the search index and handle Redis connection (async)."""

        schema = SemanticRouterIndexSchema.from_params(
            self.name, self.vectorizer.dims, self.vectorizer.dtype  # type: ignore
        )

        self._index = AsyncSearchIndex(
            schema=schema,
            redis_client=redis_client,
            redis_url=redis_url,
            **connection_kwargs,
        )

        # Check for existing router index
        existed = await self._index.exists()
        if not overwrite and existed:
            existing_index = await AsyncSearchIndex.from_existing(
                self.name, redis_client=self._index.client
            )
            if existing_index.schema.to_dict() != self._index.schema.to_dict():
                raise ValueError(
                    f"Existing index {self.name} schema does not match the user provided schema for the semantic router. "
                    "If you wish to overwrite the index schema, set overwrite=True during initialization."
                )
        await self._index.create(overwrite=overwrite, drop=False)

        if not existed or overwrite:
            # write the routes to Redis
            await self._add_routes(self.routes)

    @property
    def route_names(self) -> List[str]:
        """Get the list of route names.

        Returns:
            List[str]: List of route names.
        """
        return [route.name for route in self.routes]

    @property
    def route_thresholds(self) -> Dict[str, Optional[float]]:
        """Get the distance thresholds for each route.

        Returns:
            Dict[str, float]: Dictionary of route names and their distance thresholds.
        """
        return {route.name: route.distance_threshold for route in self.routes}

    def update_routing_config(self, routing_config: RoutingConfig):
        """Update the routing configuration.

        Args:
            routing_config (RoutingConfig): The new routing configuration.
        """
        self.routing_config = routing_config

    def update_route_thresholds(self, route_thresholds: Dict[str, Optional[float]]):
        """Update the distance thresholds for each route.

        Args:
            route_thresholds (Dict[str, float]): Dictionary of route names and their distance thresholds.
        """
        for route in self.routes:
            if route.name in route_thresholds:
                route.distance_threshold = route_thresholds[route.name]  # type: ignore

    @staticmethod
    def _route_ref_key(
        index: AsyncSearchIndex, route_name: str, reference_hash: str
    ) -> str:
        """Generate the route reference key using the index's key_separator."""
        sep = index.key_separator
        # Normalize prefix to avoid double separators
        prefix = index.prefix.rstrip(sep) if sep and index.prefix else index.prefix
        if prefix:
            return f"{prefix}{sep}{route_name}{sep}{reference_hash}"
        else:
            return f"{route_name}{sep}{reference_hash}"

    @staticmethod
    def _route_pattern(index: AsyncSearchIndex, route_name: str) -> str:
        """Generate a search pattern for route references."""
        sep = index.key_separator
        # Normalize prefix to avoid double separators
        prefix = index.prefix.rstrip(sep) if sep and index.prefix else index.prefix
        if prefix:
            return f"{prefix}{sep}{route_name}{sep}*"
        else:
            return f"{route_name}{sep}*"

    async def _add_routes(self, routes: List[Route]):
        """Add routes to the router and index (async).

        Args:
            routes (List[Route]): List of routes to be added.
        """
        route_references: List[Dict[str, Any]] = []
        keys: List[str] = []

        for route in routes:
            # embed route references as a single batch
            reference_vectors = await self.vectorizer.aembed_many(
                [reference for reference in route.references], as_buffer=True
            )
            # set route references
            for i, reference in enumerate(route.references):
                reference_hash = hashify(reference)
                route_references.append(
                    {
                        "reference_id": reference_hash,
                        "route_name": route.name,
                        "reference": reference,
                        "vector": reference_vectors[i],
                    }
                )
                keys.append(
                    self._route_ref_key(self._index, route.name, reference_hash)
                )

            # set route if does not yet exist client side
            if not self.get(route.name):
                self.routes.append(route)

        await self._index.load(route_references, keys=keys)

    def get(self, route_name: str) -> Optional[Route]:
        """Get a route by its name.

        Args:
            route_name (str): Name of the route.

        Returns:
            Optional[Route]: The selected Route object or None if not found.
        """
        return next((route for route in self.routes if route.name == route_name), None)

    def _process_route(self, result: Dict[str, Any]) -> RouteMatch:
        """Process resulting route objects and metadata."""
        route_dict = make_dict(convert_bytes(result))
        route_name = route_dict["route_name"]
        distance = float(route_dict["distance"])

        # Get route to extract model and metadata if available
        route = self.get(route_name)

        return RouteMatch(
            name=route_name,
            distance=distance,
            model=route.model if route and route.model else None,
            confidence=(1 - distance / 2) if route and route.model else None,
            metadata=route.metadata if route else {},
        )

    def _apply_cost_optimization(self, matches: List[RouteMatch]) -> List[RouteMatch]:
        """Re-rank matches considering cost when cost_optimization is enabled.

        Adjusts match distances by adding a cost penalty based on the
        cost_per_1k_input metadata field and cost_weight configuration.

        Args:
            matches: List of route matches to re-rank.

        Returns:
            Re-ranked list of matches (or original if cost optimization disabled).
        """
        if not matches or not self.routing_config.cost_optimization:
            return matches

        cost_weight = self.routing_config.cost_weight
        ranked = []

        for match in matches:
            cost = match.metadata.get("cost_per_1k_input", 0)
            # Add cost penalty to distance (normalized)
            adjusted_distance = match.distance + (cost * cost_weight)  # type: ignore
            ranked.append((match, adjusted_distance))

        # Re-sort by adjusted distance
        ranked.sort(key=lambda x: x[1])

        # Update alternatives with original matches
        for i, (match, _) in enumerate(ranked):
            match.alternatives = [
                (m.name, m.distance) for j, (m, _) in enumerate(ranked) if j != i
            ]

        return [m for m, _ in ranked]

    def _distance_threshold_filter(self) -> str:
        """Apply distance threshold on a route by route basis."""
        filter = ""
        for i, route in enumerate(self.routes):
            filter_str = f"(@route_name == '{route.name}' && @distance < {route.distance_threshold})"
            if i > 0:
                filter += " || "
            filter += filter_str

        return filter

    def _build_aggregate_request(
        self,
        vector_range_query: VectorRangeQuery,
        aggregation_method: DistanceAggregationMethod,
        max_k: int,
    ) -> AggregateRequest:
        """Build the Redis aggregation request."""
        aggregation_func: Type[Reducer]

        if aggregation_method == DistanceAggregationMethod.min:
            aggregation_func = reducers.min
        elif aggregation_method == DistanceAggregationMethod.sum:
            aggregation_func = reducers.sum
        else:
            aggregation_func = reducers.avg

        aggregate_query = str(vector_range_query).split(" RETURN")[0]
        aggregate_request = (
            AggregateRequest(aggregate_query)
            .group_by(
                "@route_name",  # type: ignore
                aggregation_func("vector_distance").alias("distance"),
            )
            .sort_by("@distance", max=max_k)  # type: ignore
            .dialect(2)
        )

        filter = self._distance_threshold_filter()

        aggregate_request.filter(filter)

        return aggregate_request

    async def _get_route_matches(
        self,
        vector: List[float],
        aggregation_method: DistanceAggregationMethod,
        max_k: int = 1,
    ) -> List[RouteMatch]:
        """Get route response from vector db (async)"""
        if not self.routes:
            return []

        # what's interesting about this is that we only provide one distance_threshold for a range query not multiple
        # therefore you might take the max_threshold and further refine from there.
        distance_threshold = max(route.distance_threshold for route in self.routes)

        vector_range_query = VectorRangeQuery(
            vector=vector,
            vector_field_name=ROUTE_VECTOR_FIELD_NAME,
            distance_threshold=float(distance_threshold),
            return_fields=["route_name"],
        )

        aggregate_request = self._build_aggregate_request(
            vector_range_query, aggregation_method, max_k=max_k
        )

        try:
            aggregation_result: AggregateResult = await self._index.aggregate(
                aggregate_request, vector_range_query.params
            )
        except ResponseError as e:
            if "VSS is not yet supported on FT.AGGREGATE" in str(e):
                raise RuntimeError(
                    "Semantic routing is only available on Redis version 7.x.x or greater"
                )
            raise e

        # process aggregation results into route matches
        return [
            self._process_route(route_match) for route_match in aggregation_result.rows
        ]

    async def _classify_route(
        self,
        vector: List[float],
        aggregation_method: DistanceAggregationMethod,
    ) -> RouteMatch:
        """Classify to a single route using a vector (async)."""

        # Get all potential route matches
        route_matches = await self._get_route_matches(
            vector, aggregation_method, max_k=len(self.routes)
        )

        if not route_matches:
            # Return default route if configured
            if self.routing_config.default_route:
                route = self.get(self.routing_config.default_route)
                if route:
                    return RouteMatch(
                        name=route.name,
                        model=route.model,
                        metadata=route.metadata,
                    )
            return RouteMatch()

        # Apply cost optimization if enabled
        route_matches = self._apply_cost_optimization(route_matches)

        # Take top route after optimization
        top_route_match = route_matches[0]

        if top_route_match.name is not None:
            return top_route_match
        else:
            raise ValueError(
                f"{top_route_match.name} not a supported route for the {self.name} semantic router."
            )

    async def _classify_multi_route(
        self,
        vector: List[float],
        max_k: int,
        aggregation_method: DistanceAggregationMethod,
    ) -> List[RouteMatch]:
        """Classify to multiple routes, up to max_k (int), using a vector (async)."""

        route_matches = await self._get_route_matches(
            vector, aggregation_method, max_k=max_k
        )

        # Apply cost optimization if enabled
        route_matches = self._apply_cost_optimization(route_matches)

        # Process route matches
        top_route_matches: List[RouteMatch] = []
        if route_matches:
            for route_match in route_matches:
                if route_match.name is not None:
                    top_route_matches.append(route_match)
                else:
                    raise ValueError(
                        f"{route_match.name} not a supported route for the {self.name} semantic router."
                    )

        return top_route_matches

    @deprecated_argument("distance_threshold")
    async def __call__(
        self,
        statement: Optional[str] = None,
        vector: Optional[List[float]] = None,
        aggregation_method: Optional[DistanceAggregationMethod] = None,
        distance_threshold: Optional[float] = None,
    ) -> RouteMatch:
        """Query the semantic router with a given statement or vector (async).

        Args:
            statement (Optional[str]): The input statement to be queried.
            vector (Optional[List[float]]): The input vector to be queried.
            distance_threshold (Optional[float]): The threshold for semantic distance.
            aggregation_method (Optional[DistanceAggregationMethod]): The aggregation method used for vector distances.

        Returns:
            RouteMatch: The matching route.
        """
        if not vector:
            if not statement:
                raise ValueError("Must provide a vector or statement to the router")
            vector = await self.vectorizer.aembed(statement)

        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        # perform route classification
        top_route_match = await self._classify_route(vector, aggregation_method)  # type: ignore
        return top_route_match

    @deprecated_argument("distance_threshold")
    async def route_many(
        self,
        statement: Optional[str] = None,
        vector: Optional[List[float]] = None,
        max_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        aggregation_method: Optional[DistanceAggregationMethod] = None,
    ) -> List[RouteMatch]:
        """Query the semantic router with a given statement or vector for multiple matches (async).

        Args:
            statement (Optional[str]): The input statement to be queried.
            vector (Optional[List[float]]): The input vector to be queried.
            max_k (Optional[int]): The maximum number of top matches to return.
            distance_threshold (Optional[float]): The threshold for semantic distance.
            aggregation_method (Optional[DistanceAggregationMethod]): The aggregation method used for vector distances.

        Returns:
            List[RouteMatch]: The matching routes and their details.
        """
        if not vector:
            if not statement:
                raise ValueError("Must provide a vector or statement to the router")
            vector = await self.vectorizer.aembed(statement)

        max_k = max_k or self.routing_config.max_k
        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        # classify routes
        top_route_matches = await self._classify_multi_route(
            vector, max_k, aggregation_method  # type: ignore
        )

        return top_route_matches

    async def remove_route(self, route_name: str) -> None:
        """Remove a route and all references from the semantic router (async).

        Args:
            route_name (str): Name of the route to remove.
        """
        route = self.get(route_name)
        if route is None:
            logger.warning(f"Route {route_name} is not found in the SemanticRouter")
        else:
            await self._index.drop_keys(
                [
                    self._route_ref_key(self._index, route.name, hashify(reference))
                    for reference in route.references
                ]
            )
            self.routes = [route for route in self.routes if route.name != route_name]

    async def delete(self) -> None:
        """Delete the semantic router index and associated config (async)."""
        await self._index.delete(drop=True)
        # Clean up router config key
        await self._index.client.delete(f"{self.name}:route_config")  # type: ignore

    async def clear(self) -> None:
        """Flush all routes from the semantic router index (async)."""
        await self._index.clear()
        self.routes = []

    @classmethod
    async def from_dict(
        cls,
        data: Dict[str, Any],
        **kwargs,
    ) -> "AsyncSemanticRouter":
        """Create an AsyncSemanticRouter from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary containing the semantic router data.

        Returns:
            AsyncSemanticRouter: The async semantic router instance.

        Raises:
            ValueError: If required data is missing or invalid.
        """
        from redisvl.utils.vectorize import vectorizer_from_dict

        try:
            name = data["name"]
            routes_data = data["routes"]
            vectorizer_data = data["vectorizer"]
            routing_config_data = data["routing_config"]
        except KeyError as e:
            raise ValueError(f"Unable to load semantic router from dict: {str(e)}")

        try:
            vectorizer = vectorizer_from_dict(vectorizer_data)
        except Exception as e:
            raise ValueError(f"Unable to load vectorizer: {str(e)}")

        if not vectorizer:
            raise ValueError(f"Unable to load vectorizer: {vectorizer_data}")

        routes = [Route(**route) for route in routes_data]
        routing_config = RoutingConfig(**routing_config_data)

        return await cls.create(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the AsyncSemanticRouter instance to a dictionary.

        Returns:
            Dict[str, Any]: The dictionary representation of the AsyncSemanticRouter.
        """
        return {
            "name": self.name,
            "routes": [model_to_dict(route) for route in self.routes],
            "vectorizer": {
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            "routing_config": model_to_dict(self.routing_config),
        }

    @classmethod
    async def from_yaml(
        cls,
        file_path: str,
        **kwargs,
    ) -> "AsyncSemanticRouter":
        """Create an AsyncSemanticRouter from a YAML file.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            AsyncSemanticRouter: The async semantic router instance.

        Raises:
            ValueError: If the file path is invalid.
            FileNotFoundError: If the file does not exist.
        """
        try:
            fp = Path(file_path).resolve()
        except OSError as e:
            raise ValueError(f"Invalid file path: {file_path}") from e

        if not fp.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        with open(fp, "r") as f:
            yaml_data = yaml.safe_load(f)
            return await cls.from_dict(
                yaml_data,
                **kwargs,
            )

    @classmethod
    async def from_pretrained(
        cls,
        config_name_or_path: str,
        redis_client: Optional[AsyncRedisClient] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = True,
        **kwargs,
    ) -> "AsyncSemanticRouter":
        """Load router from pretrained config with embeddings (async).

        This skips the embedding step by using pre-computed vectors.
        Accepts either a file path or a built-in config name (e.g., "default").

        Args:
            config_name_or_path: Path to pretrained JSON file, or name of a
                built-in config (e.g., "default").
            redis_client: Async Redis client.
            redis_url: Redis URL.
            overwrite: Whether to overwrite an existing index with the same
                name. Defaults to True.
            **kwargs: Additional connection arguments.

        Returns:
            AsyncSemanticRouter loaded without re-embedding.
        """
        import json

        import numpy as np

        from redisvl.extensions.router.pretrained import get_pretrained_path
        from redisvl.extensions.router.schema import PretrainedRouterConfig
        from redisvl.utils.vectorize import vectorizer_from_dict

        fp = Path(config_name_or_path)
        if not fp.exists():
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
            await RedisConnectionFactory.validate_async_redis(redis_client)
        elif redis_url:
            redis_client = await RedisConnectionFactory._get_aredis_connection(
                redis_url=redis_url, **kwargs
            )

        if redis_client is None:
            raise ValueError("Could not establish Redis connection")

        # Create index schema
        schema = SemanticRouterIndexSchema.from_params(
            config.name, vectorizer.dims, vectorizer.dtype  # type: ignore
        )

        index = AsyncSearchIndex(
            schema=schema,
            redis_client=redis_client,
        )
        await index.create(overwrite=overwrite, drop=False)

        # Load pre-computed embeddings directly
        routes = []
        all_references = []
        all_keys = []

        for pr in config.routes:
            route = Route(
                name=pr.name,
                references=[r.text for r in pr.references],
                metadata=pr.metadata,
                distance_threshold=pr.distance_threshold,
                model=pr.model,
            )
            routes.append(route)

            # Use pre-computed vectors (convert to buffer)
            for ref in pr.references:
                vector_buffer = np.array(ref.vector, dtype=np.float32).tobytes()
                reference_hash = hashify(ref.text)

                sep = index.key_separator
                prefix = (
                    index.prefix.rstrip(sep) if sep and index.prefix else index.prefix
                )
                if prefix:
                    key = f"{prefix}{sep}{pr.name}{sep}{reference_hash}"
                else:
                    key = f"{pr.name}{sep}{reference_hash}"

                all_references.append(
                    {
                        "reference_id": reference_hash,
                        "route_name": pr.name,
                        "reference": ref.text,
                        "vector": vector_buffer,
                    }
                )
                all_keys.append(key)

        await index.load(all_references, keys=all_keys)

        # Create router instance using Pydantic's model_construct to bypass __init__
        routing_config = RoutingConfig(**config.routing_config)

        router = cls.model_construct(
            name=config.name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
        )
        # Set private attribute directly
        object.__setattr__(router, "_index", index)

        # Store config
        await redis_client.json().set(f"{config.name}:route_config", ".", router.to_dict())  # type: ignore

        return router

    def export_with_embeddings(self, file_path: str):
        """Export router with pre-computed embeddings.

        This allows loading the router without re-embedding references.

        Args:
            file_path: Path to JSON file.
        """
        import json

        from redisvl.extensions.router.schema import (
            PretrainedReference,
            PretrainedRoute,
            PretrainedRouterConfig,
        )

        fp = Path(file_path).resolve()

        pretrained_routes = []
        for route in self.routes:
            # Get embeddings for all references
            vectors = self.vectorizer.embed_many(route.references)

            references = [
                PretrainedReference(text=ref, vector=vec)
                for ref, vec in zip(route.references, vectors)
            ]

            pretrained_routes.append(
                PretrainedRoute(
                    name=route.name,
                    references=references,
                    metadata=route.metadata,
                    distance_threshold=route.distance_threshold,
                    model=route.model,
                )
            )

        config = PretrainedRouterConfig(
            name=self.name,
            vectorizer={
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            routes=pretrained_routes,
            routing_config=model_to_dict(self.routing_config),
        )

        with open(fp, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

    def to_yaml(self, file_path: str, overwrite: bool = True) -> None:
        """Write the semantic router to a YAML file.

        Args:
            file_path (str): The path to the YAML file.
            overwrite (bool): Whether to overwrite the file if it already exists.

        Raises:
            FileExistsError: If the file already exists and overwrite is False.
        """
        fp = Path(file_path).resolve()
        if fp.exists() and not overwrite:
            raise FileExistsError(f"Schema file {file_path} already exists.")

        with open(fp, "w") as f:
            yaml_data = self.to_dict()
            yaml.dump(yaml_data, f, sort_keys=False)

    # reference methods
    async def add_route_references(
        self,
        route_name: str,
        references: Union[str, List[str]],
    ) -> List[str]:
        """Add a reference(s) to an existing route (async).

        Args:
            router_name (str): The name of the router.
            references (Union[str, List[str]]): The reference or list of references to add.

        Returns:
            List[str]: The list of added references keys.
        """

        if isinstance(references, str):
            references = [references]

        route_references: List[Dict[str, Any]] = []
        keys: List[str] = []

        # embed route references as a single batch
        reference_vectors = await self.vectorizer.aembed_many(
            references, as_buffer=True
        )

        # set route references
        for i, reference in enumerate(references):
            reference_hash = hashify(reference)

            route_references.append(
                {
                    "reference_id": reference_hash,
                    "route_name": route_name,
                    "reference": reference,
                    "vector": reference_vectors[i],
                }
            )
            keys.append(self._route_ref_key(self._index, route_name, reference_hash))

        keys = await self._index.load(route_references, keys=keys)

        route = self.get(route_name)
        if not route:
            raise ValueError(f"Route {route_name} not found in the SemanticRouter")
        route.references.extend(references)
        await self._update_router_state()
        return keys

    @staticmethod
    def _make_filter_queries(ids: List[str]) -> List[FilterQuery]:
        """Create a filter query for the given ids."""

        queries = []

        for id in ids:
            fe = Tag("reference_id") == id
            fq = FilterQuery(
                return_fields=["reference_id", "route_name", "reference"],
                filter_expression=fe,
            )
            queries.append(fq)

        return queries

    async def get_route_references(
        self,
        route_name: str = "",
        reference_ids: List[str] = [],
        keys: List[str] = [],
    ) -> List[Dict[str, Any]]:
        """Get references for an existing route route (async).

        Args:
            router_name (str): The name of the router.
            references (Union[str, List[str]]): The reference or list of references to add.

        Returns:
            List[Dict[str, Any]]]: Reference objects stored
        """

        if reference_ids:
            queries = self._make_filter_queries(reference_ids)
        elif route_name:
            if not keys:
                pattern = self._route_pattern(self._index, route_name)
                client = self._index.client
                keys_list = []
                async for key in client.scan_iter(match=pattern):  # type: ignore
                    keys_list.append(key)
                keys = keys_list

            sep = self._index.key_separator
            queries = self._make_filter_queries(
                [key.split(sep)[-1] for key in convert_bytes(keys)]
            )
        else:
            raise ValueError(
                "Must provide a route name, reference ids, or keys to get references"
            )

        res = await self._index.batch_query(queries)  # type: ignore

        return [r[0] for r in res if len(r) > 0]

    async def delete_route_references(
        self,
        route_name: str = "",
        reference_ids: List[str] = [],
        keys: List[str] = [],
    ) -> int:
        """Get references for an existing semantic router route (async).

        Args:
            router_name Optional(str): The name of the router.
            reference_ids Optional(List[str]]): The reference or list of references to delete.
            keys Optional(List[str]]): List of fully qualified keys (prefix:router:reference_id) to delete.

        Returns:
            int: Number of objects deleted
        """

        if reference_ids and not keys:
            queries = self._make_filter_queries(reference_ids)
            res = await self._index.batch_query(queries)  # type: ignore
            keys = [r[0]["id"] for r in res if len(r) > 0]
        elif not keys:
            pattern = self._route_pattern(self._index, route_name)
            client = self._index.client
            keys_list = []
            async for key in client.scan_iter(match=pattern):  # type: ignore
                keys_list.append(key)
            keys = keys_list

        if not keys:
            raise ValueError(f"No references found for route {route_name}")

        to_be_deleted = []
        client = self._index.client
        for key in keys:
            route_name = key.split(":")[-2]
            hgetall_result = await client.hgetall(key)  # type: ignore
            to_be_deleted.append((route_name, convert_bytes(hgetall_result)))

        deleted = await self._index.drop_keys(keys)

        for route_name, delete in to_be_deleted:
            route = self.get(route_name)
            if not route:
                raise ValueError(f"Route {route_name} not found in the SemanticRouter")
            route.references.remove(delete["reference"])

        await self._update_router_state()

        return deleted

    async def _update_router_state(self) -> None:
        """Update the router configuration in Redis (async)."""
        client = self._index.client
        await client.json().set(f"{self.name}:route_config", ".", self.to_dict())  # type: ignore
