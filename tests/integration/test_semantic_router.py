import pytest

from redisvl.extensions.router import SemanticRouter
from redisvl.extensions.router.schema import Route, RoutingConfig
from redisvl.redis.connection import compare_versions


@pytest.fixture
def routes():
    return [
        Route(
            name="greeting",
            references=["hello", "hi"],
            metadata={"type": "greeting"},
            distance_threshold=0.3,
        ),
        Route(
            name="farewell",
            references=["bye", "goodbye"],
            metadata={"type": "farewell"},
            distance_threshold=0.3,
        ),
    ]


@pytest.fixture
def semantic_router(client, routes):
    router = SemanticRouter(
        name="test-router",
        routes=routes,
        routing_config=RoutingConfig(distance_threshold=0.3, max_k=2),
        redis_client=client,
        overwrite=False,
    )
    yield router
    router.delete()


def test_initialize_router(semantic_router):
    assert semantic_router.name == "test-router"
    assert len(semantic_router.routes) == 2
    assert semantic_router.routing_config.distance_threshold == 0.3
    assert semantic_router.routing_config.max_k == 2


def test_router_properties(semantic_router):
    route_names = semantic_router.route_names
    assert "greeting" in route_names
    assert "farewell" in route_names

    thresholds = semantic_router.route_thresholds
    assert thresholds["greeting"] == 0.3
    assert thresholds["farewell"] == 0.3


def test_get_route(semantic_router):
    route = semantic_router.get("greeting")
    assert route is not None
    assert route.name == "greeting"
    assert "hello" in route.references


def test_get_non_existing_route(semantic_router):
    route = semantic_router.get("non_existent_route")
    assert route is None


def test_single_query(semantic_router):
    redis_version = semantic_router._index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.0.0"):
        pytest.skip("Not using a late enough version of Redis")

    match = semantic_router("hello")
    assert match.name == "greeting"
    assert match.distance <= semantic_router.route_thresholds["greeting"]


def test_single_query_no_match(semantic_router):
    redis_version = semantic_router._index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.0.0"):
        pytest.skip("Not using a late enough version of Redis")

    match = semantic_router("unknown_phrase")
    assert match.name is None


def test_multiple_query(semantic_router):
    redis_version = semantic_router._index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.0.0"):
        pytest.skip("Not using a late enough version of Redis")

    matches = semantic_router.route_many("hello", max_k=2)
    assert len(matches) > 0
    assert matches[0].name == "greeting"


def test_update_routing_config(semantic_router):
    new_config = RoutingConfig(distance_threshold=0.5, max_k=1)
    semantic_router.update_routing_config(new_config)
    assert semantic_router.routing_config.distance_threshold == 0.5
    assert semantic_router.routing_config.max_k == 1


def test_vector_query(semantic_router):
    redis_version = semantic_router._index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.0.0"):
        pytest.skip("Not using a late enough version of Redis")

    vector = semantic_router.vectorizer.embed("goodbye")
    match = semantic_router(vector=vector)
    assert match.name == "farewell"


def test_vector_query_no_match(semantic_router):
    redis_version = semantic_router._index.client.info()["redis_version"]
    if not compare_versions(redis_version, "7.0.0"):
        pytest.skip("Not using a late enough version of Redis")

    vector = [
        0.0
    ] * semantic_router.vectorizer.dims  # Random vector unlikely to match any route
    match = semantic_router(vector=vector)
    assert match.name is None


def test_add_route(semantic_router):
    new_routes = [
        Route(
            name="politics",
            references=[
                "are you liberal or conservative?",
                "who will you vote for?",
                "political speech",
            ],
            metadata={"type": "greeting"},
        )
    ]
    semantic_router._add_routes(new_routes)

    route = semantic_router.get("politics")
    assert route is not None
    assert route.name == "politics"
    assert "political speech" in route.references

    redis_version = semantic_router._index.client.info()["redis_version"]
    if compare_versions(redis_version, "7.0.0"):
        match = semantic_router("political speech")
        print(match, flush=True)
        assert match is not None
        assert match.name == "politics"


def test_remove_routes(semantic_router):
    semantic_router.remove_route("greeting")
    assert semantic_router.get("greeting") is None

    semantic_router.remove_route("unknown_route")
    assert semantic_router.get("unknown_route") is None
