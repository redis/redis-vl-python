import hashlib
from typing import Any, Dict, List, Optional, Type

import redis.commands.search.reducers as reducers
from pydantic.v1 import BaseModel, Field, PrivateAttr
from redis import Redis
from redis.commands.search.aggregation import AggregateRequest, AggregateResult, Reducer
from redis.exceptions import ResponseError

from redisvl.extensions.router.schema import (
    DistanceAggregationMethod,
    Route,
    RouteMatch,
    RoutingConfig,
)
from redisvl.index import SearchIndex
from redisvl.query import RangeQuery
from redisvl.redis.utils import convert_bytes, hashify, make_dict
from redisvl.schema import IndexInfo, IndexSchema
from redisvl.utils.log import get_logger
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer

logger = get_logger(__name__)


class SemanticRouterIndexSchema(IndexSchema):
    """Customized index schema for SemanticRouter."""

    @classmethod
    def from_params(cls, name: str, vector_dims: int) -> "SemanticRouterIndexSchema":
        """Create an index schema based on router name and vector dimensions.

        Args:
            name (str): The name of the index.
            vector_dims (int): The dimensions of the vectors.

        Returns:
            SemanticRouterIndexSchema: The constructed index schema.
        """
        return cls(
            index=IndexInfo(name=name, prefix=name),
            fields=[  # type: ignore
                {"name": "route_name", "type": "tag"},
                {"name": "reference", "type": "text"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": vector_dims,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        )


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

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        name: str,
        routes: List[Route],
        vectorizer: Optional[BaseVectorizer] = None,
        routing_config: Optional[RoutingConfig] = None,
        redis_client: Optional[Redis] = None,
        redis_url: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """Initialize the SemanticRouter.

        Args:
            name (str): The name of the semantic router.
            routes (List[Route]): List of Route objects.
            vectorizer (BaseVectorizer, optional): The vectorizer used to embed route references. Defaults to default HFTextVectorizer.
            routing_config (RoutingConfig, optional): Configuration for routing behavior. Defaults to the default RoutingConfig.
            redis_client (Optional[Redis], optional): Redis client for connection. Defaults to None.
            redis_url (Optional[str], optional): Redis URL for connection. Defaults to None.
            overwrite (bool, optional): Whether to overwrite existing index. Defaults to False.
            **kwargs: Additional arguments.
        """
        # Set vectorizer default
        if vectorizer is None:
            vectorizer = HFTextVectorizer()

        if routing_config is None:
            routing_config = RoutingConfig()

        super().__init__(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
        )
        self._initialize_index(redis_client, redis_url, overwrite)

    def _initialize_index(
        self,
        redis_client: Optional[Redis] = None,
        redis_url: Optional[str] = None,
        overwrite: bool = False,
        **connection_kwargs,
    ):
        """Initialize the search index and handle Redis connection."""
        schema = SemanticRouterIndexSchema.from_params(self.name, self.vectorizer.dims)
        self._index = SearchIndex(schema=schema)

        if redis_client:
            self._index.set_client(redis_client)
        else:
            self._index.connect(redis_url=redis_url, **connection_kwargs)

        existed = self._index.exists()
        self._index.create(overwrite=overwrite)

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

    def _route_ref_key(self, route_name: str, reference: str) -> str:
        """Generate the route reference key."""
        reference_hash = hashify(reference)
        return f"{self._index.prefix}:{route_name}:{reference_hash}"

    def _add_routes(self, routes: List[Route]):
        """Add routes to the router and index.

        Args:
            routes (List[Route]): List of routes to be added.
        """
        route_references: List[Dict[str, Any]] = []
        keys: List[str] = []

        for route in routes:
            # set route references
            for reference in route.references:
                route_references.append(
                    {
                        "route_name": route.name,
                        "reference": reference,
                        "vector": self.vectorizer.embed(reference, as_buffer=True),
                    }
                )
                keys.append(self._route_ref_key(route.name, reference))

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
        return RouteMatch(
            name=route_dict["route_name"], distance=float(route_dict["distance"])
        )

    def _build_aggregate_request(
        self,
        vector_range_query: RangeQuery,
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
                "@route_name", aggregation_func("vector_distance").alias("distance")
            )
            .sort_by("@distance", max=max_k)
            .dialect(2)
        )

        return aggregate_request

    def _classify_route(
        self,
        vector: List[float],
        distance_threshold: float,
        aggregation_method: DistanceAggregationMethod,
    ) -> RouteMatch:
        """Classify to a single route using a vector."""
        vector_range_query = RangeQuery(
            vector=vector,
            vector_field_name="vector",
            distance_threshold=distance_threshold,
            return_fields=["route_name"],
        )

        aggregate_request = self._build_aggregate_request(
            vector_range_query, aggregation_method, max_k=1
        )

        try:
            aggregation_result: AggregateResult = self._index.client.ft(  # type: ignore
                self._index.name
            ).aggregate(aggregate_request, vector_range_query.params)
        except ResponseError as e:
            if "VSS is not yet supported on FT.AGGREGATE" in str(e):
                raise RuntimeError(
                    "Semantic routing is only available on Redis version 7.x.x or greater"
                )
            raise e

        # process aggregation results into route matches
        route_matches = [
            self._process_route(route_match) for route_match in aggregation_result.rows
        ]

        # process route matches
        if route_matches:
            top_route_match = route_matches[0]
            if top_route_match.name is not None:
                if route := self.get(top_route_match.name):
                    # use the matched route's distance threshold
                    _distance_threshold = route.distance_threshold or distance_threshold
                    if self._pass_threshold(top_route_match, _distance_threshold):
                        return top_route_match
                else:
                    raise ValueError(
                        f"{top_route_match.name} not a supported route for the {self.name} semantic router."
                    )

        # fallback to empty route match if no hits
        return RouteMatch()

    def _classify_multi_route(
        self,
        vector: List[float],
        max_k: int,
        distance_threshold: float,
        aggregation_method: DistanceAggregationMethod,
    ) -> List[RouteMatch]:
        """Classify to multiple routes, up to max_k (int), using a vector."""
        vector_range_query = RangeQuery(
            vector=vector,
            vector_field_name="vector",
            distance_threshold=distance_threshold,
            return_fields=["route_name"],
        )
        aggregate_request = self._build_aggregate_request(
            vector_range_query, aggregation_method, max_k
        )

        try:
            aggregation_result: AggregateResult = self._index.client.ft(  # type: ignore
                self._index.name
            ).aggregate(aggregate_request, vector_range_query.params)
        except ResponseError as e:
            if "VSS is not yet supported on FT.AGGREGATE" in str(e):
                raise RuntimeError(
                    "Semantic routing is only available on Redis version 7.x.x or greater"
                )
            raise e

        # process aggregation results into route matches
        route_matches = [
            self._process_route(route_match) for route_match in aggregation_result.rows
        ]

        # process route matches
        top_route_matches: List[RouteMatch] = []
        if route_matches:
            for route_match in route_matches:
                if route_match.name is not None:
                    if route := self.get(route_match.name):
                        # use the matched route's distance threshold
                        _distance_threshold = (
                            route.distance_threshold or distance_threshold
                        )
                        if self._pass_threshold(route_match, _distance_threshold):
                            top_route_matches.append(route_match)
                    else:
                        raise ValueError(
                            f"{route_match.name} not a supported route for the {self.name} semantic router."
                        )

        return top_route_matches

    def _pass_threshold(
        self, route_match: Optional[RouteMatch], distance_threshold: float
    ) -> bool:
        """Check if a route match passes the distance threshold.

        Args:
            route_match (Optional[RouteMatch]): The route match to check.
            distance_threshold (float): The fallback distance threshold to use if not assigned to a route.

        Returns:
            bool: True if the route match passes the threshold, False otherwise.
        """
        if route_match and distance_threshold:
            if route_match.distance is not None:
                return route_match.distance <= distance_threshold
        return False

    def __call__(
        self,
        statement: Optional[str] = None,
        vector: Optional[List[float]] = None,
        distance_threshold: Optional[float] = None,
        aggregation_method: Optional[DistanceAggregationMethod] = None,
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
            vector = self.vectorizer.embed(statement)

        # override routing config
        distance_threshold = (
            distance_threshold or self.routing_config.distance_threshold
        )
        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        # perform route classification
        top_route_match = self._classify_route(
            vector, distance_threshold, aggregation_method
        )
        return top_route_match

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
            vector = self.vectorizer.embed(statement)

        # override routing config defaults
        distance_threshold = (
            distance_threshold or self.routing_config.distance_threshold
        )
        max_k = max_k or self.routing_config.max_k
        aggregation_method = (
            aggregation_method or self.routing_config.aggregation_method
        )

        # classify routes
        top_route_matches = self._classify_multi_route(
            vector, max_k, distance_threshold, aggregation_method
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
                    self._route_ref_key(route.name, reference)
                    for reference in route.references
                ]
            )
            self.routes = [route for route in self.routes if route.name != route_name]

    def delete(self) -> None:
        """Delete the semantic router index."""
        self._index.delete(drop=True)

    def clear(self) -> None:
        """Flush all routes from the semantic router index."""
        self._index.clear()
        self.routes = []
