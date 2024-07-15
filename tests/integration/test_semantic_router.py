import pytest
from redisvl.extensions.router.routes import Route, RoutingConfig
from redisvl.extensions.router.semantic import SemanticRouter


@pytest.fixture
def routes():
    politics = Route(
        name="politics",
        references=[
            "isn't politics the best thing ever",
            "why don't you tell me about your political opinions"
        ],
        metadata={"priority": "1"}
    )
    chitchat = Route(
        name="chitchat",
        references=[
            "hello",
            "how's the weather today?",
            "how are things going?"
        ],
        metadata={"priority": "2"}
    )
    return [politics, chitchat]

@pytest.fixture
def semantic_router(redis_client, routes):
    config = RoutingConfig(distance_threshold=1.0)
    router = SemanticRouter(
        name="topic-router",
        routes=routes,
        routing_config=config,
        redis_client=redis_client,
        overwrite=True
    )
    return router


def test_semantic_router_match_politics(semantic_router):
    result = semantic_router("I am thinking about running for Governor in the state of VA. What do I need to consider?")
    assert result[0]['route'].name == "politics"


def test_semantic_router_match_chitchat(semantic_router):
    result = semantic_router("hello")
    assert result[0]['route'].name == "chitchat"


def test_semantic_router_no_match(semantic_router):
    result = semantic_router("unrelated topic")
    assert result == []


def test_update_routing_config(semantic_router):
    new_config = RoutingConfig(distance_threshold=0.1, sort_by='avg_distance')

    semantic_router.update_routing_config(new_config)
    result = semantic_router("hello world")
    assert result == []

    result = semantic_router("hello world", distance_threshold=0.3)
    assert len(result) > 0
