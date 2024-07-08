from pydantic.v1 import BaseModel, root_validator, Field
from typing import Any, List, Dict, Optional, Union
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema, IndexInfo
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer
from redisvl.extensions.router.routes import Route, RoutingConfig

import hashlib


class SemanticRouterIndexSchema(IndexSchema):

    @classmethod
    def from_params(cls, name: str, vector_dims: int):
        return cls(
            index=IndexInfo(name=name, prefix=name),
            fields={
                "route_name": {"name": "route_name", "type": "tag"},
                "reference": {"name": "reference", "type": "text"},
                "vector": {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": vector_dims,
                        "distance_metric": "cosine",
                        "datatype": "float32"
                    }
                }
            }
        )


class SemanticRouter(BaseModel):
    name: str
    """The name of the semantic router"""
    vectorizer: BaseVectorizer = Field(default_factory=HFTextVectorizer)
    """The vectorizer used to embed route references"""
    routes: List[Route]
    """List of Route objects"""
    routing_config: RoutingConfig = Field(default_factory=RoutingConfig)
    """Configuration for routing behavior"""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_index(**data)

    def _initialize_index(self, **data):
        """Initialize the search index and handle Redis connection.

        Args:
            data (dict): Initialization data containing Redis connection details.
        """
        # Extract connection parameters
        redis_url = data.pop("redis_url", "redis://localhost:6379")
        redis_client = data.pop("redis_client", None)
        connection_args = data.pop("connection_args", {})

        # Create search index schema
        schema = SemanticRouterIndexSchema.from_params(self.name, self.vectorizer.dims)

        # Build search index
        self._index = SearchIndex(schema=schema)

        # Handle Redis connection
        if redis_client:
            self._index.set_client(redis_client)
        else:
            self._index.connect(redis_url=redis_url, **connection_args)

        if not self._index.exists():
            self._add_routes(self.routes)

        self._index.create(overwrite=False)

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

        for route in routes:
            for reference in route.references:
                route_references.append({
                    "route_name": route.name,
                    "reference": reference,
                    "vector": self.vectorizer.embed(reference)
                })
                reference_hash = hashlib.sha256(reference.encode("utf-8")).hexdigest()
                keys.append(f"{self._index.schema.index.prefix}:{route.name}:{reference_hash}")

        self._index.load(route_references, keys=keys)


    def __call__(
        self,
        statement: str,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query the semantic router with a given statement.

        Args:
            statement (str): The input statement to be queried.
            top_k (Optional[int]): The maximum number of top matches to return.
            distance_threshold (Optional[float]): The threshold for semantic distance.

        Returns:
            List[Dict[str, Any]]: The matching routes and their details.
        """
        vector = self.vectorizer.embed(statement)
        top_k = top_k if top_k is not None else self.routing_config.top_k
        distance_threshold = distance_threshold if distance_threshold is not None else self.routing_config.distance_threshold

        # TODO: Implement the query logic based on top_k and distance_threshold
        results = []

        return results
