"""Async integration tests for AsyncLLMRouter.

Mirrors sync LLMRouter tests with async fixtures and await calls.
"""

import pathlib

import pytest
from ulid import ULID

from redisvl.extensions.llm_router import AsyncLLMRouter, LLMRouteMatch, ModelTier
from redisvl.extensions.llm_router.schema import RoutingConfig
from tests.conftest import (
    SKIP_HF,
    skip_if_no_redisearch_async,
    skip_if_redis_version_below_async,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        SKIP_HF, reason="sentence-transformers not supported on Python 3.14+"
    ),
]


def get_base_path():
    return pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def model_tiers():
    """Define model tiers for testing."""
    return [
        ModelTier(
            name="simple",
            model="openai/gpt-4.1-nano",
            references=[
                "hello",
                "hi there",
                "what time is it?",
                "thanks",
                "goodbye",
            ],
            metadata={
                "provider": "openai",
                "cost_per_1k_input": 0.0001,
                "cost_per_1k_output": 0.0004,
            },
            distance_threshold=0.5,
        ),
        ModelTier(
            name="reasoning",
            model="anthropic/claude-sonnet-4-5",
            references=[
                "analyze this code for bugs",
                "explain how neural networks learn",
                "write a detailed blog post about",
                "compare and contrast these approaches",
                "debug this issue step by step",
            ],
            metadata={
                "provider": "anthropic",
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.015,
            },
            distance_threshold=0.6,
        ),
        ModelTier(
            name="expert",
            model="anthropic/claude-opus-4-5",
            references=[
                "prove this mathematical theorem",
                "architect a distributed system for millions of users",
                "write a research paper analyzing",
                "review this legal contract for issues",
                "design a novel algorithm for",
            ],
            metadata={
                "provider": "anthropic",
                "cost_per_1k_input": 0.005,
                "cost_per_1k_output": 0.025,
            },
            distance_threshold=0.7,
        ),
    ]


@pytest.fixture
async def async_llm_router(async_client, model_tiers, hf_vectorizer):
    """Create an AsyncLLMRouter for testing."""
    await skip_if_no_redisearch_async(async_client)
    router = await AsyncLLMRouter.create(
        name=f"test-async-llm-router-{str(ULID())}",
        tiers=model_tiers,
        vectorizer=hf_vectorizer,
        redis_client=async_client,
        overwrite=True,
    )
    yield router
    await router.delete()


class TestAsyncLLMRouterInitialization:
    """Test AsyncLLMRouter initialization and configuration."""

    async def test_initialize_router(self, async_llm_router):
        """Router should initialize with tiers."""
        assert async_llm_router.name.startswith("test-async-llm-router-")
        assert len(async_llm_router.tiers) == 3
        assert async_llm_router.tier_names == ["simple", "reasoning", "expert"]

    async def test_router_properties(self, async_llm_router):
        """Router should expose tier properties."""
        tier_names = async_llm_router.tier_names
        assert "simple" in tier_names
        assert "reasoning" in tier_names
        assert "expert" in tier_names

        thresholds = async_llm_router.tier_thresholds
        assert thresholds["simple"] == 0.5
        assert thresholds["reasoning"] == 0.6
        assert thresholds["expert"] == 0.7

    async def test_get_tier(self, async_llm_router):
        """Should retrieve tier by name."""
        tier = async_llm_router.get_tier("simple")
        assert tier is not None
        assert tier.name == "simple"
        assert tier.model == "openai/gpt-4.1-nano"

    async def test_get_nonexistent_tier(self, async_llm_router):
        """Should return None for nonexistent tier."""
        tier = async_llm_router.get_tier("nonexistent")
        assert tier is None


class TestAsyncLLMRouterRouting:
    """Test AsyncLLMRouter routing behavior."""

    async def test_route_simple_query(self, async_llm_router, async_client):
        """Simple greetings should route to simple tier."""
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        match = await async_llm_router.route("hello, how are you?")
        assert isinstance(match, LLMRouteMatch)
        assert match.tier == "simple"
        assert match.model == "openai/gpt-4.1-nano"
        assert match.distance is not None
        assert match.distance <= 0.5

    async def test_route_reasoning_query(self, async_llm_router, async_client):
        """Complex analysis should route to reasoning tier."""
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        match = await async_llm_router.route(
            "analyze this code and find potential bugs"
        )
        assert match.tier == "reasoning"
        assert match.model == "anthropic/claude-sonnet-4-5"

    async def test_route_expert_query(self, async_llm_router, async_client):
        """Research-level queries should route to expert tier."""
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        match = await async_llm_router.route(
            "design a novel distributed consensus algorithm"
        )
        assert match.tier == "expert"
        assert match.model == "anthropic/claude-opus-4-5"

    async def test_route_no_match(self, async_llm_router, async_client):
        """Unrelated queries should return default tier or no match."""
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        match = await async_llm_router.route("xyzzy plugh random gibberish 12345")
        assert match.tier is None or match.tier == async_llm_router.default_tier

    async def test_route_with_vector(self, async_llm_router, async_client):
        """Should accept pre-computed vector."""
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        vector = async_llm_router.vectorizer.embed("hello")
        match = await async_llm_router.route(vector=vector)
        assert match.tier == "simple"

    async def test_route_many(self, async_llm_router, async_client):
        """Should return multiple tier matches."""
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        matches = await async_llm_router.route_many(
            "explain machine learning concepts", max_k=3
        )
        assert len(matches) > 0
        assert all(isinstance(m, LLMRouteMatch) for m in matches)


class TestAsyncLLMRouterCostOptimization:
    """Test cost-aware routing behavior."""

    async def test_cost_optimization_prefers_cheaper(
        self, async_client, model_tiers, hf_vectorizer
    ):
        """With cost optimization, should prefer cheaper tiers when close."""
        await skip_if_no_redisearch_async(async_client)
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        router = await AsyncLLMRouter.create(
            name=f"test-async-cost-router-{str(ULID())}",
            tiers=model_tiers,
            vectorizer=hf_vectorizer,
            redis_client=async_client,
            cost_optimization=True,
            overwrite=True,
        )

        try:
            match = await router.route("hello there, how are you?")
            assert match.tier is not None or router.default_tier is not None
        finally:
            await router.delete()


class TestAsyncLLMRouterSerialization:
    """Test AsyncLLMRouter serialization and persistence."""

    async def test_to_dict(self, async_llm_router):
        """Should serialize to dictionary."""
        router_dict = async_llm_router.to_dict()
        assert router_dict["name"] == async_llm_router.name
        assert len(router_dict["tiers"]) == 3
        assert router_dict["vectorizer"]["type"] == async_llm_router.vectorizer.type

    async def test_from_dict(self, async_llm_router):
        """Should deserialize from dictionary."""
        router_dict = async_llm_router.to_dict()
        new_router = await AsyncLLMRouter.from_dict(
            router_dict,
            redis_client=async_llm_router._index.client,
            overwrite=True,
        )
        assert new_router.name == async_llm_router.name
        assert len(new_router.tiers) == len(async_llm_router.tiers)

    async def test_to_yaml(self, async_llm_router, tmp_path):
        """Should serialize to YAML file."""
        yaml_file = tmp_path / "async_llm_router.yaml"
        async_llm_router.to_yaml(str(yaml_file))
        assert yaml_file.exists()

    async def test_from_yaml(self, async_llm_router, tmp_path):
        """Should deserialize from YAML file."""
        yaml_file = tmp_path / "async_llm_router.yaml"
        async_llm_router.to_yaml(str(yaml_file))

        new_router = await AsyncLLMRouter.from_yaml(
            str(yaml_file),
            redis_client=async_llm_router._index.client,
            overwrite=True,
        )
        assert new_router.name == async_llm_router.name


class TestAsyncLLMRouterWithEmbeddings:
    """Test AsyncLLMRouter with pre-computed embeddings."""

    async def test_export_with_embeddings(self, async_llm_router, tmp_path):
        """Should export router with pre-computed embeddings."""
        json_file = tmp_path / "async_router_with_embeddings.json"
        await async_llm_router.export_with_embeddings(str(json_file))

        assert json_file.exists()

        import json

        with open(json_file) as f:
            data = json.load(f)

        assert "tiers" in data
        for tier in data["tiers"]:
            assert "references" in tier
            for ref in tier["references"]:
                assert "text" in ref
                assert "vector" in ref
                assert isinstance(ref["vector"], list)
                assert len(ref["vector"]) > 0

    async def test_import_with_embeddings(self, async_client, hf_vectorizer, tmp_path):
        """Should import router without re-embedding."""
        await skip_if_no_redisearch_async(async_client)
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        tier = ModelTier(
            name="test",
            model="test/model",
            references=["hello", "world"],
            distance_threshold=0.5,
        )
        router1 = await AsyncLLMRouter.create(
            name=f"async-export-test-{str(ULID())}",
            tiers=[tier],
            vectorizer=hf_vectorizer,
            redis_client=async_client,
            overwrite=True,
        )

        json_file = tmp_path / "async_export_test.json"
        await router1.export_with_embeddings(str(json_file))
        await router1.delete()

        router2 = await AsyncLLMRouter.from_pretrained(
            str(json_file),
            redis_client=async_client,
        )

        try:
            assert len(router2.tiers) == 1
            assert router2.tiers[0].name == "test"

            match = await router2.route("hello there")
            assert match.tier == "test"
        finally:
            await router2.delete()


class TestAsyncLLMRouterTierManagement:
    """Test adding/removing tiers."""

    async def test_add_tier(self, async_llm_router, async_client):
        """Should add new tier."""
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        new_tier = ModelTier(
            name="local",
            model="ollama/llama3.2",
            references=["ok", "yes", "no"],
            metadata={"cost_per_1k_input": 0},
            distance_threshold=0.3,
        )
        await async_llm_router.add_tier(new_tier)

        assert async_llm_router.get_tier("local") is not None
        assert "local" in async_llm_router.tier_names

    async def test_remove_tier(self, async_llm_router):
        """Should remove tier."""
        await async_llm_router.remove_tier("simple")

        assert async_llm_router.get_tier("simple") is None
        assert "simple" not in async_llm_router.tier_names

    async def test_add_tier_references(self, async_llm_router, async_client):
        """Should add references to existing tier."""
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        match_before = await async_llm_router.route("hi there")
        assert match_before.tier == "simple"

        await async_llm_router.add_tier_references(
            tier_name="simple",
            references=["howdy", "greetings friend"],
        )

        tier = async_llm_router.get_tier("simple")
        assert "howdy" in tier.references
        assert "greetings friend" in tier.references

    async def test_update_tier_threshold(self, async_llm_router):
        """Should update tier threshold."""
        await async_llm_router.update_tier_threshold("simple", 0.3)

        assert async_llm_router.tier_thresholds["simple"] == 0.3


class TestAsyncLLMRouterFromExisting:
    """Test reconnecting to existing router."""

    async def test_from_existing(
        self, async_client, model_tiers, hf_vectorizer, redis_url
    ):
        """Should reconnect to existing router."""
        await skip_if_no_redisearch_async(async_client)
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        router_name = f"async-persist-test-{str(ULID())}"

        router1 = await AsyncLLMRouter.create(
            name=router_name,
            tiers=model_tiers,
            vectorizer=hf_vectorizer,
            redis_url=redis_url,
            overwrite=True,
        )

        router2 = await AsyncLLMRouter.from_existing(
            name=router_name,
            redis_url=redis_url,
        )

        try:
            assert router2.name == router1.name
            assert len(router2.tiers) == len(router1.tiers)
            assert router2.tier_names == router1.tier_names

            match = await router2.route("hello")
            assert match.tier is not None
        finally:
            await router1.delete()


class TestAsyncLLMRouterPretrained:
    """Test AsyncLLMRouter with pretrained configurations."""

    async def test_from_pretrained_default(self, async_client):
        """Should load the built-in default pretrained config."""
        await skip_if_no_redisearch_async(async_client)

        router = await AsyncLLMRouter.from_pretrained(
            "default", redis_client=async_client
        )
        try:
            assert len(router.tiers) == 3
            assert set(router.tier_names) == {"simple", "standard", "expert"}
        finally:
            await router.delete()

    async def test_pretrained_routes_simple(self, async_client):
        """Simple greetings should route to simple tier."""
        await skip_if_no_redisearch_async(async_client)
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        router = await AsyncLLMRouter.from_pretrained(
            "default", redis_client=async_client
        )
        try:
            match = await router.route("hi, how are you doing?")
            assert match.tier == "simple"
            assert match.model == "openai/gpt-4.1-nano"
        finally:
            await router.delete()

    async def test_pretrained_routes_expert(self, async_client):
        """Complex research queries should route to expert tier."""
        await skip_if_no_redisearch_async(async_client)
        await skip_if_redis_version_below_async(async_client, "7.0.0")

        router = await AsyncLLMRouter.from_pretrained(
            "default", redis_client=async_client
        )
        try:
            match = await router.route("architect a fault-tolerant distributed system")
            assert match.tier == "expert"
            assert match.model == "anthropic/claude-opus-4-5"
        finally:
            await router.delete()

    async def test_pretrained_invalid_name(self, async_client):
        """Should raise error for unknown pretrained config name."""
        with pytest.raises(ValueError, match="not found"):
            await AsyncLLMRouter.from_pretrained(
                "nonexistent_config", redis_client=async_client
            )
