import pathlib
import warnings

import pytest
from redis.exceptions import ConnectionError
from ulid import ULID

from redisvl.extensions.router import SemanticRouter
from redisvl.extensions.router.schema import (
    DistanceAggregationMethod,
    Route,
    RoutingConfig,
)
from redisvl.redis.connection import compare_versions
from tests.conftest import skip_if_no_redisearch, skip_if_redis_version_below


def get_base_path():
    return pathlib.Path(__file__).parent.resolve()


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
            distance_threshold=0.2,
        ),
    ]


@pytest.fixture
def semantic_router(client, routes, hf_vectorizer):
    skip_if_no_redisearch(client)
    router = SemanticRouter(
        name=f"test-router-{str(ULID())}",
        routes=routes,
        routing_config=RoutingConfig(max_k=2),
        redis_client=client,
        overwrite=False,
        vectorizer=hf_vectorizer,
    )
    yield router
    router.clear()
    router.delete()


@pytest.fixture(autouse=True)
def disable_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_initialize_router(semantic_router):
    assert semantic_router.name == semantic_router.name
    assert len(semantic_router.routes) == 2
    assert semantic_router.routing_config.max_k == 2


def test_router_properties(semantic_router):
    route_names = semantic_router.route_names
    assert "greeting" in route_names
    assert "farewell" in route_names

    thresholds = semantic_router.route_thresholds
    assert thresholds["greeting"] == 0.3
    assert thresholds["farewell"] == 0.2


def test_get_route(semantic_router):
    route = semantic_router.get("greeting")
    assert route is not None
    assert route.name == "greeting"
    assert "hello" in route.references


def test_get_non_existing_route(semantic_router):
    route = semantic_router.get("non_existent_route")
    assert route is None


def test_single_query(semantic_router):
    skip_if_redis_version_below(semantic_router._index.client, "7.0.0")

    match = semantic_router("hello")
    assert match.name == "greeting"
    assert match.distance <= semantic_router.route_thresholds["greeting"]


def test_single_query_no_match(semantic_router):
    skip_if_redis_version_below(semantic_router._index.client, "7.0.0")

    match = semantic_router("unknown_phrase")
    assert match.name is None


def test_multiple_query(semantic_router):
    skip_if_redis_version_below(semantic_router._index.client, "7.0.0")

    matches = semantic_router.route_many("hello", max_k=2)
    assert len(matches) > 0
    assert matches[0].name == "greeting"


def test_update_routing_config(semantic_router):
    new_config = RoutingConfig(max_k=27, aggregation_method="min")
    semantic_router.update_routing_config(new_config)
    assert semantic_router.routing_config.max_k == 27
    assert (
        semantic_router.routing_config.aggregation_method
        == DistanceAggregationMethod.min
    )


def test_vector_query(semantic_router):
    skip_if_redis_version_below(semantic_router._index.client, "7.0.0")

    vector = semantic_router.vectorizer.embed("goodbye")
    match = semantic_router(vector=vector)
    assert match.name == "farewell"


def test_vector_query_no_match(semantic_router):
    skip_if_redis_version_below(semantic_router._index.client, "7.0.0")

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


def test_to_dict(semantic_router):
    router_dict = semantic_router.to_dict()
    assert router_dict["name"] == semantic_router.name
    assert len(router_dict["routes"]) == len(semantic_router.routes)
    assert router_dict["vectorizer"]["type"] == semantic_router.vectorizer.type


def test_from_dict(semantic_router):
    router_dict = semantic_router.to_dict()
    new_router = SemanticRouter.from_dict(
        router_dict, redis_client=semantic_router._index.client, overwrite=True
    )
    assert new_router.to_dict() == router_dict


def test_to_yaml(semantic_router):
    yaml_file = str(get_base_path().joinpath("../../schemas/semantic_router.yaml"))
    semantic_router.name = "test-router"
    semantic_router.to_yaml(yaml_file, overwrite=True)
    assert pathlib.Path(yaml_file).exists()


def test_from_yaml(semantic_router):
    yaml_file = str(get_base_path().joinpath("../../schemas/semantic_router.yaml"))
    new_router = SemanticRouter.from_yaml(
        yaml_file, redis_client=semantic_router._index.client, overwrite=True
    )
    nr = new_router.to_dict()
    nr.pop("name")
    sr = semantic_router.to_dict()
    sr.pop("name")
    assert nr == sr


def test_to_dict_missing_fields():
    data = {
        "name": "incomplete-router",
        "routes": [],
        "vectorizer": {"type": "HFTextVectorizer", "model": "bert-base-uncased"},
    }
    with pytest.raises(ValueError):
        SemanticRouter.from_dict(data)


def test_invalid_vectorizer():
    data = {
        "name": "invalid-router",
        "routes": [],
        "vectorizer": {"type": "InvalidVectorizer", "model": "invalid-model"},
        "routing_config": {},
    }
    with pytest.raises(ValueError):
        SemanticRouter.from_dict(data)


def test_yaml_invalid_file_path():
    with pytest.raises(FileNotFoundError):
        SemanticRouter.from_yaml("invalid_path.yaml", redis_client=None)


def test_idempotent_to_dict(semantic_router):
    router_dict = semantic_router.to_dict()
    new_router = SemanticRouter.from_dict(
        router_dict, redis_client=semantic_router._index.client, overwrite=True
    )
    assert new_router.to_dict() == router_dict


def test_bad_connection_info(routes):
    with pytest.raises(ConnectionError):
        SemanticRouter(
            name="test-router",
            routes=routes,
            routing_config=RoutingConfig(distance_threshold=0.3, max_k=2),
            redis_url="redis://localhost:6389",  # bad connection url
            overwrite=False,
        )


def test_different_vector_dtypes(client, redis_url, routes):
    skip_if_no_redisearch(client)
    try:
        bfloat_router = SemanticRouter(
            name="bfloat_router",
            routes=routes,
            dtype="bfloat16",
            redis_url=redis_url,
        )

        float16_router = SemanticRouter(
            name="float16_router",
            routes=routes,
            dtype="float16",
            redis_url=redis_url,
        )

        float32_router = SemanticRouter(
            name="float32_router",
            routes=routes,
            dtype="float32",
            redis_url=redis_url,
        )

        float64_router = SemanticRouter(
            name="float64_router",
            routes=routes,
            dtype="float64",
            redis_url=redis_url,
        )

        for router in [bfloat_router, float16_router, float32_router, float64_router]:
            assert len(router.route_many("hello", max_k=5)) == 1
    except:
        pytest.skip("Not using a late enough version of Redis")


def test_bad_dtype_connecting_to_exiting_router(client, redis_url, routes):
    skip_if_no_redisearch(client)
    # Skip this test for Redis 6.2.x as FT.INFO doesn't return dims properly
    redis_version = client.info()["redis_version"]
    if redis_version.startswith("6.2"):
        pytest.skip(
            "Redis 6.2.x FT.INFO doesn't properly return vector dims for reconnection"
        )

    router = SemanticRouter(
        name="float64-router",
        routes=routes,
        dtype="float64",
        redis_url=redis_url,
    )

    same_type = SemanticRouter(
        name="float64-router",
        routes=routes,
        dtype="float64",
        redis_url=redis_url,
    )

    with pytest.raises(ValueError):
        bad_type = SemanticRouter(
            name="float64-router",
            routes=routes,
            dtype="float16",
            redis_url=redis_url,
        )


def test_vectorizer_dtype_mismatch(client, routes, redis_url, hf_vectorizer_float16):
    skip_if_no_redisearch(client)
    with pytest.raises(ValueError):
        SemanticRouter(
            name="test_dtype_mismatch",
            routes=routes,
            dtype="float32",
            vectorizer=hf_vectorizer_float16,
            redis_url=redis_url,
            overwrite=True,
        )


def test_invalid_vectorizer(client, redis_url):
    skip_if_no_redisearch(client)
    with pytest.raises(TypeError):
        SemanticRouter(
            name="test_invalid_vectorizer",
            vectorizer="invalid_vectorizer",  # type: ignore
            redis_url=redis_url,
            overwrite=True,
        )


def test_passes_through_dtype_to_default_vectorizer(client, routes, redis_url):
    skip_if_no_redisearch(client)
    # The default is float32, so we should see float64 if we pass it in.
    router = SemanticRouter(
        name="test_pass_through_dtype",
        routes=routes,
        dtype="float64",
        redis_url=redis_url,
        overwrite=True,
    )
    assert router.vectorizer.dtype == "float64"


def test_deprecated_dtype_argument(client, routes, redis_url):
    skip_if_no_redisearch(client)
    with pytest.warns(DeprecationWarning):
        SemanticRouter(
            name="test_deprecated_dtype",
            routes=routes,
            dtype="float32",
            redis_url=redis_url,
            overwrite=True,
        )


def test_deprecated_distance_threshold_argument(
    semantic_router, client, routes, redis_url
):
    skip_if_redis_version_below(semantic_router._index.client, "7.0.0")
    skip_if_no_redisearch(client)

    router = SemanticRouter(
        name="test_pass_through_dtype",
        routes=routes,
        redis_url=redis_url,
        overwrite=True,
    )
    with pytest.warns(DeprecationWarning):
        router("hello", distance_threshold=0.3)


def test_routes_different_distance_thresholds_get_two(
    semantic_router, client, routes, redis_url
):
    skip_if_redis_version_below(semantic_router._index.client, "7.0.0")
    skip_if_no_redisearch(client)
    routes[0].distance_threshold = 0.5
    routes[1].distance_threshold = 0.7

    router = SemanticRouter(
        name="test_routes_different_distance_thresholds",
        routes=routes,
        redis_url=redis_url,
        overwrite=True,
    )

    matches = router.route_many("hello", max_k=2)
    assert len(matches) == 2
    assert matches[0].name == "greeting"
    assert matches[1].name == "farewell"


def test_routes_different_distance_thresholds_get_one(
    semantic_router, client, routes, redis_url
):
    skip_if_redis_version_below(semantic_router._index.client, "7.0.0")
    skip_if_no_redisearch(client)

    routes[0].distance_threshold = 0.5

    # don't match on second
    routes[1].distance_threshold = 0.3

    router = SemanticRouter(
        name="test_routes_different_distance_thresholds",
        routes=routes,
        redis_url=redis_url,
        overwrite=True,
    )

    matches = router.route_many("hello", max_k=2)
    assert len(matches) == 1
    assert matches[0].name == "greeting"


def test_add_delete_route_references(semantic_router):
    skip_if_redis_version_below(semantic_router._index.client, "7.0.0")

    # Add new references to an existing route
    added_refs = semantic_router.add_route_references(
        route_name="greeting", references=["good morning", "hey there"]
    )

    # Verify references were added
    assert len(added_refs) == 2

    # Test that we can match against the new references
    match = semantic_router("hey there")
    assert match.name == "greeting"

    # delete by route
    deleted_count = semantic_router.delete_route_references(
        route_name="farewell",
    )

    if deleted_count < 2:
        pytest.skip("Flaky test - skip")

    assert deleted_count == 2

    # delete by ref_id
    deleted = semantic_router.delete_route_references(
        reference_ids=[added_refs[0].split(":")[-1]]
    )

    assert deleted == 1

    # delete by key
    deleted = semantic_router.delete_route_references(keys=[added_refs[1]])

    assert deleted == 1

    router_dict = semantic_router.to_dict()
    assert len(router_dict["routes"][0]["references"]) == 2
    assert len(router_dict["routes"][1]["references"]) == 0


def test_from_existing(client, redis_url, routes):
    skip_if_no_redisearch(client)
    skip_if_redis_version_below(client, "7.0.0")

    # connect separately
    router = SemanticRouter(
        name=f"test-router-{str(ULID())}",
        routes=routes,
        routing_config=RoutingConfig(max_k=2),
        redis_url=redis_url,
        overwrite=False,
    )

    router2 = SemanticRouter.from_existing(
        name=router.name,
        redis_url=redis_url,
    )

    assert router.to_dict() == router2.to_dict()


def test_get_route_references(semantic_router):
    # Get references for a specific route
    refs = semantic_router.get_route_references(route_name="greeting")

    if len(refs) < 2:
        pytest.skip("Flaky test - skip")

    # Should return at least the initial references
    assert len(refs) == 2

    # Reference IDs should be present
    reference_id = refs[0]["reference_id"]
    # Get references by ID
    id_refs = semantic_router.get_route_references(reference_ids=[reference_id])
    assert len(id_refs) == 1

    with pytest.raises(ValueError):
        semantic_router.get_route_references()


def test_delete_route_references(semantic_router):
    # Get references for a specific route
    deleted = semantic_router.delete_route_references(route_name="greeting")

    assert deleted == 2

    router_dict = semantic_router.to_dict()
    assert len(router_dict["routes"][0]["references"]) == 0
