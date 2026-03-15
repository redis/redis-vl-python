"""Unit tests for LLM routing schema extensions."""

import pytest
from pydantic import ValidationError

from redisvl.extensions.router.schema import (
    DistanceAggregationMethod,
    PretrainedReference,
    PretrainedRoute,
    PretrainedRouterConfig,
    Route,
    RouteMatch,
    RoutingConfig,
)


class TestRouteWithModel:
    """Tests for Route with optional model field."""

    def test_route_with_model(self):
        """Should create route with model."""
        route = Route(
            name="simple",
            model="openai/gpt-4.1-nano",
            references=["hello", "hi"],
            distance_threshold=0.5,
        )
        assert route.name == "simple"
        assert route.model == "openai/gpt-4.1-nano"
        assert route.references == ["hello", "hi"]
        assert route.distance_threshold == 0.5

    def test_route_without_model(self):
        """Model should be optional."""
        route = Route(
            name="generic",
            references=["hello"],
        )
        assert route.model is None

    def test_route_with_metadata(self):
        """Should accept metadata."""
        route = Route(
            name="simple",
            model="test/model",
            references=["hello"],
            metadata={
                "cost_per_1k_input": 0.0001,
                "capabilities": ["chat"],
            },
        )
        assert route.metadata["cost_per_1k_input"] == 0.0001
        assert "chat" in route.metadata["capabilities"]

    def test_empty_name_fails(self):
        """Should reject empty name."""
        with pytest.raises(ValidationError):
            Route(
                name="",
                model="test/model",
                references=["hello"],
            )

    def test_empty_references_fails(self):
        """Should reject empty references."""
        with pytest.raises(ValidationError):
            Route(
                name="test",
                model="test/model",
                references=[],
            )

    def test_whitespace_reference_fails(self):
        """Should reject whitespace-only references."""
        with pytest.raises(ValidationError):
            Route(
                name="test",
                model="test/model",
                references=["hello", "  "],
            )

    def test_threshold_bounds(self):
        """Should validate threshold bounds (0, 2]."""
        Route(name="t", model="m", references=["r"], distance_threshold=0.1)
        Route(name="t", model="m", references=["r"], distance_threshold=2.0)

        with pytest.raises(ValidationError):
            Route(name="t", model="m", references=["r"], distance_threshold=0)

        with pytest.raises(ValidationError):
            Route(name="t", model="m", references=["r"], distance_threshold=-0.1)

        with pytest.raises(ValidationError):
            Route(name="t", model="m", references=["r"], distance_threshold=2.1)


class TestRouteMatch:
    """Tests for RouteMatch schema."""

    def test_empty_match(self):
        """Empty match should be falsy."""
        match = RouteMatch()
        assert not match
        assert match.name is None
        assert match.model is None

    def test_valid_match(self):
        """Valid match should be truthy."""
        match = RouteMatch(
            name="simple",
            model="test/model",
            distance=0.3,
            confidence=0.85,
        )
        assert match
        assert match.name == "simple"
        assert match.confidence == 0.85

    def test_match_with_alternatives(self):
        """Should store alternative matches."""
        match = RouteMatch(
            name="simple",
            model="test/model",
            distance=0.3,
            alternatives=[("reasoning", 0.5), ("expert", 0.7)],
        )
        assert len(match.alternatives) == 2
        assert match.alternatives[0] == ("reasoning", 0.5)

    def test_match_with_metadata(self):
        """Should store route metadata."""
        match = RouteMatch(
            name="simple",
            model="test/model",
            metadata={"cost_per_1k_input": 0.0001},
        )
        assert match.metadata["cost_per_1k_input"] == 0.0001


class TestRoutingConfig:
    """Tests for RoutingConfig schema."""

    def test_defaults(self):
        """Should have sensible defaults."""
        config = RoutingConfig()
        assert config.max_k == 1
        assert config.aggregation_method == DistanceAggregationMethod.avg
        assert config.cost_optimization is False
        assert config.cost_weight == 0.1
        assert config.default_route is None

    def test_custom_config(self):
        """Should accept custom values."""
        config = RoutingConfig(
            max_k=3,
            aggregation_method=DistanceAggregationMethod.min,
            cost_optimization=True,
            cost_weight=0.5,
            default_route="simple",
        )
        assert config.max_k == 3
        assert config.aggregation_method == DistanceAggregationMethod.min
        assert config.cost_optimization is True
        assert config.default_route == "simple"

    def test_cost_weight_bounds(self):
        """Cost weight should be 0-1."""
        RoutingConfig(cost_weight=0)
        RoutingConfig(cost_weight=1)

        with pytest.raises(ValidationError):
            RoutingConfig(cost_weight=-0.1)

        with pytest.raises(ValidationError):
            RoutingConfig(cost_weight=1.1)

    def test_max_k_positive(self):
        """max_k should be positive."""
        with pytest.raises(ValidationError):
            RoutingConfig(max_k=0)

        with pytest.raises(ValidationError):
            RoutingConfig(max_k=-1)


class TestPretrainedSchemas:
    """Tests for pretrained configuration schemas."""

    def test_pretrained_reference(self):
        """Should store text and vector."""
        ref = PretrainedReference(
            text="hello",
            vector=[0.1, 0.2, 0.3],
        )
        assert ref.text == "hello"
        assert ref.vector == [0.1, 0.2, 0.3]

    def test_pretrained_route(self):
        """Should store route with embedded references."""
        route = PretrainedRoute(
            name="simple",
            model="test/model",
            references=[
                PretrainedReference(text="hello", vector=[0.1, 0.2]),
                PretrainedReference(text="hi", vector=[0.3, 0.4]),
            ],
            distance_threshold=0.5,
        )
        assert route.name == "simple"
        assert len(route.references) == 2
        assert route.references[0].text == "hello"

    def test_pretrained_router_config(self):
        """Should store complete pretrained config."""
        config = PretrainedRouterConfig(
            name="test-router",
            version="1.0.0",
            vectorizer={"type": "hf", "model": "test-model"},
            routes=[
                PretrainedRoute(
                    name="simple",
                    model="test/model",
                    references=[
                        PretrainedReference(text="hello", vector=[0.1]),
                    ],
                )
            ],
        )
        assert config.name == "test-router"
        assert config.version == "1.0.0"
        assert len(config.routes) == 1


class TestDistanceAggregationMethod:
    """Tests for aggregation method enum."""

    def test_values(self):
        """Should have expected values."""
        assert DistanceAggregationMethod.avg.value == "avg"
        assert DistanceAggregationMethod.min.value == "min"
        assert DistanceAggregationMethod.sum.value == "sum"

    def test_from_string(self):
        """Should parse from string."""
        assert DistanceAggregationMethod("avg") == DistanceAggregationMethod.avg
        assert DistanceAggregationMethod("min") == DistanceAggregationMethod.min
