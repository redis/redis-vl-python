from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import redis.commands.search.reducers as reducers
import yaml
from pydantic.v1 import BaseModel, Field, PrivateAttr
from redis import Redis
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
from redisvl.index import SearchIndex
from redisvl.query import RangeQuery
from redisvl.redis.utils import convert_bytes, hashify, make_dict
from redisvl.utils.log import get_logger
from redisvl.utils.utils import model_to_dict
from redisvl.utils.vectorize import (
    BaseVectorizer,
    HFTextVectorizer,
    vectorizer_from_dict,
)

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

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        name: str,
        routes: List[Route],
        vectorizer: Optional[BaseVectorizer] = None,
        routing_config: Optional[RoutingConfig] = None,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        connection_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """Initialize the SemanticRouter.

        Args:
            name (str): The name of the semantic router.
            routes (List[Route]): List of Route objects.
            vectorizer (BaseVectorizer, optional): The vectorizer used to embed route references. Defaults to default HFTextVectorizer.
            routing_config (RoutingConfig, optional): Configuration for routing behavior. Defaults to the default RoutingConfig.
            redis_client (Optional[Redis], optional): Redis client for connection. Defaults to None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            overwrite (bool, optional): Whether to overwrite existing index. Defaults to False.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.
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
        dtype = kwargs.get("dtype", "float32")
        self._initialize_index(
            redis_client, redis_url, overwrite, dtype, **connection_kwargs
        )

    def _initialize_index(
        self,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        dtype: str = "float32",
        **connection_kwargs,
    ):
        """Initialize the search index and handle Redis connection."""
        schema = SemanticRouterIndexSchema.from_params(
            self.name, self.vectorizer.dims, dtype
        )
        self._index = SearchIndex(schema=schema)

        if redis_client:
            self._index.set_client(redis_client)
        elif redis_url:
            self._index.connect(redis_url=redis_url, **connection_kwargs)

        # Check for existing router index
        existed = self._index.exists()
        if not overwrite and existed:
            existing_index = SearchIndex.from_existing(
                self.name, redis_client=self._index.client
            )
            if existing_index.schema != self._index.schema:
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
            # embed route references as a single batch
            reference_vectors = self.vectorizer.embed_many(
                [reference for reference in route.references],
                as_buffer=True,
                dtype=self._index.schema.fields[ROUTE_VECTOR_FIELD_NAME].attrs.datatype,  # type: ignore[union-attr]
            )
            # set route references
            for i, reference in enumerate(route.references):
                route_references.append(
                    {
                        "route_name": route.name,
                        "reference": reference,
                        "vector": reference_vectors[i],
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
            vector_field_name=ROUTE_VECTOR_FIELD_NAME,
            distance_threshold=distance_threshold,
            return_fields=["route_name"],
            dtype=self._index.schema.fields[ROUTE_VECTOR_FIELD_NAME].attrs.datatype,  # type: ignore[union-attr]
        )

        aggregate_request = self._build_aggregate_request(
            vector_range_query, aggregation_method, max_k=1
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
            vector_field_name=ROUTE_VECTOR_FIELD_NAME,
            distance_threshold=distance_threshold,
            return_fields=["route_name"],
            dtype=self._index.schema.fields[ROUTE_VECTOR_FIELD_NAME].attrs.datatype,  # type: ignore[union-attr]
        )
        aggregate_request = self._build_aggregate_request(
            vector_range_query, aggregation_method, max_k
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
