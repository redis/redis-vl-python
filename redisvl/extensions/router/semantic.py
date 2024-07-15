from pydantic.v1 import BaseModel, root_validator, Field, PrivateAttr
from typing import Any, List, Dict, Optional, Union

from redis import Redis
from redis.commands.search.aggregation import AggregateRequest, AggregateResult, Reducer
import redis.commands.search.reducers as reducers

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, RangeQuery
from redisvl.schema import IndexSchema, IndexInfo
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer
from redisvl.extensions.router.routes import Route, RoutingConfig, AccumulationMethod

from redisvl.redis.utils import make_dict, convert_bytes

import hashlib


class SemanticRouterIndexSchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, vector_dims: int):
        return cls(
            index=IndexInfo(name=name, prefix=name),
            fields=[
                {"name": "route_name", "type": "tag"},
                {"name": "reference", "type": "text"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": vector_dims,
                        "distance_metric": "cosine",
                        "datatype": "float32"
                    }
                }
            ]
        )


class SemanticRouter(BaseModel):
    name: str
    """The name of the semantic router"""
    routes: List[Route]
    """List of Route objects"""
    vectorizer: BaseVectorizer = Field(default_factory=HFTextVectorizer)
    """The vectorizer used to embed route references"""
    routing_config: RoutingConfig = Field(default_factory=RoutingConfig)
    """Configuration for routing behavior"""

    _index: SearchIndex = PrivateAttr()
    # _accumulation_method: AccumulationMethod = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        name: str,
        routes: List[Route],
        vectorizer: BaseVectorizer = HFTextVectorizer(),
        routing_config: RoutingConfig = RoutingConfig(),
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **kwargs
    ):
        """Initialize the SemanticRouter.

        Args:
            name (str): The name of the semantic router.
            routes (List[Route]): List of Route objects.
            vectorizer (BaseVectorizer, optional): The vectorizer used to embed route references. Defaults to HFTextVectorizer().
            routing_config (RoutingConfig, optional): Configuration for routing behavior. Defaults to RoutingConfig().
            redis_client (Optional[Redis], optional): Redis client for connection. Defaults to None.
            redis_url (str, optional): Redis URL for connection. Defaults to "redis://localhost:6379".
            overwrite (bool, optional): Whether to overwrite existing index. Defaults to False.
            **kwargs: Additional arguments.
        """
        super().__init__(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config
        )
        self._initialize_index(redis_client, redis_url, overwrite)
        # self._accumulation_method = self._pick_accumulation_method()

    def _initialize_index(
        self,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **connection_kwargs
    ):
        """Initialize the search index and handle Redis connection.

        Args:
            redis_client (Optional[Redis], optional): Redis client for connection. Defaults to None.
            redis_url (str, optional): Redis URL for connection. Defaults to "redis://localhost:6379".
            overwrite (bool, optional): Whether to overwrite existing index. Defaults to False.
            **connection_kwargs: Additional connection arguments.
        """
        schema = SemanticRouterIndexSchema.from_params(self.name, self.vectorizer.dims)
        self._index = SearchIndex(schema=schema)

        if redis_client:
            self._index.set_client(redis_client)
        else:
            self._index.connect(redis_url=redis_url, **connection_kwargs)

        existed = self._index.exists()
        self._index.create(overwrite=overwrite)

        if not existed or overwrite:
            self._add_routes(self.routes)

    # def _pick_accumulation_method(self) -> AccumulationMethod:
    #     """Pick the accumulation method based on the routing configuration."""
    #     if self.routing_config.accumulation_method != AccumulationMethod.auto:
    #         return self.routing_config.accumulation_method

    #     num_route_references = [len(route.references) for route in self.routes]
    #     avg_num_references = sum(num_route_references) / len(num_route_references)
    #     variance = sum((x - avg_num_references) ** 2 for x in num_route_references) / len(num_route_references)

    #     if variance < 1:  # TODO: Arbitrary threshold for low variance
    #         return AccumulationMethod.sum
    #     else:
    #         return AccumulationMethod.avg

    def update_routing_config(self, routing_config: RoutingConfig):
        """Update the routing configuration.

        Args:
            routing_config (RoutingConfig): The new routing configuration.
        """
        self.routing_config = routing_config
        # self._accumulation_method = self._pick_accumulation_method()

    def _add_routes(self, routes: List[Route]):
        """Add routes to the index.

        Args:
            routes (List[Route]): List of routes to be added.
        """
        route_references: List[Dict[str, Any]] = []
        keys: List[str] = []

        for route in routes:
            for reference in route.references:
                route_references.append({
                    "route_name": route.name,
                    "reference": reference,
                    "vector": self.vectorizer.embed(reference, as_buffer=True)
                })
                reference_hash = hashlib.sha256(reference.encode("utf-8")).hexdigest()
                keys.append(f"{self._index.schema.index.prefix}:{route.name}:{reference_hash}")

        self._index.load(route_references, keys=keys)

    def __call__(
        self,
        statement: str,
        max_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query the semantic router with a given statement.

        Args:
            statement (str): The input statement to be queried.
            max_k (Optional[int]): The maximum number of top matches to return.
            distance_threshold (Optional[float]): The threshold for semantic distance.

        Returns:
            List[Dict[str, Any]]: The matching routes and their details.
        """
        vector = self.vectorizer.embed(statement)
        max_k = max_k if max_k is not None else self.routing_config.max_k
        distance_threshold = distance_threshold if distance_threshold is not None else self.routing_config.distance_threshold

        # # get the total number of route references in the index
        # num_route_references = sum(
        #     [len(route.references) for route in self.routes]
        # )
        # define the baseline range query to fetch relevant route references
        vector_range_query = RangeQuery(
            vector=vector,
            vector_field_name="vector",
            distance_threshold=2,
            return_fields=["route_name"]
        )

        # build redis aggregation query
        aggregate_query = str(vector_range_query).split(" RETURN")[0]
        aggregate_request = (
            AggregateRequest(aggregate_query)
                .group_by(
                    "@route_name",
                    reducers.avg("vector_distance").alias("avg"),
                    reducers.min("vector_distance").alias("score")
                )
                .apply(avg_score="1 - @avg", score="1 - @score")
                .dialect(2)
        )

        top_routes_and_scores = []
        aggregate_results = self._index.client.ft(self._index.name).aggregate(aggregate_request, vector_range_query.params)

        for result in aggregate_results.rows:
            top_routes_and_scores.append(make_dict(convert_bytes(result)))

        top_routes = self._fetch_routes(top_routes_and_scores)

        return top_routes


    def _fetch_routes(self, top_routes_and_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch route objects and metadata based on top matches.

        Args:
            top_routes_and_scores: List of top routes and their scores.

        Returns:
            List[Dict[str, Any]]: Routes with their metadata.
        """
        results = []
        for route_info in top_routes_and_scores:
            route_name = route_info["route_name"]
            route = next((r for r in self.routes if r.name == route_name), None)
            if route:
                results.append({
                    **route.dict(),
                    "score": route_info["score"],
                    "avg_score": route_info["avg_score"]
                })

        return results
