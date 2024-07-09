from pydantic.v1 import BaseModel, root_validator, Field, PrivateAttr
from typing import Any, List, Dict, Optional, Union
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, RangeQuery
from redisvl.schema import IndexSchema, IndexInfo
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer
from redisvl.extensions.router.routes import Route, RoutingConfig

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
        super().__init__(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config
        )
        self._initialize_index(redis_client, redis_url, overwrite)

    def _initialize_index(
        self,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **connection_kwargs
    ):
        """Initialize the search index and handle Redis connection.

        Args:
            data (dict): Initialization data containing Redis connection details.
        """
        # Create search index schema
        schema = SemanticRouterIndexSchema.from_params(self.name, self.vectorizer.dims)

        # Build search index
        self._index = SearchIndex(schema=schema)

        # Handle Redis connection
        if redis_client:
            self._index.set_client(redis_client)
        else:
            self._index.connect(redis_url=redis_url, **connection_kwargs)

        existed = self._index.exists()
        self._index.create(overwrite=overwrite)

        # If the index did not yet exist OR we overwrote it
        if not existed or overwrite:
            self._add_routes(self.routes)

        # TODO : double check this kind of logic



    def update_routing_config(self, routing_config: RoutingConfig):
        """Update the routing configuration.

        Args:
            routing_config (RoutingConfig): The new routing configuration.
        """
        self.routing_config = routing_config
        # TODO: Ensure Pydantic handles the validation here
        # TODO: Determine if we need to persist this to Redis

    def _add_routes(self, routes: List[Route]):
        """Add routes to the index.

        Args:
            routes (List[Route]): List of routes to be added.
        """
        route_references: List[Dict[str, Any]] = []
        keys: List[str] = []
        # Iteratively load route references
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

        query = RangeQuery(
            vector=vector,
            vector_field_name="vector",
            distance_threshold=distance_threshold,
            return_fields=["route_name", "reference"],
        )

        route_references = self._index.query(query)

        # TODO use accumulation strategy to aggregation (sum or avg) the scores by the associated route
        #top_routes_and_scores = ...

        # TODO fetch the route objects and metadata directly from this class based on top matches
        #results = ...

        return route_references
