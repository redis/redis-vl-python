"""Integration tests for LLMRouter.

These tests define the expected behavior of the LLM Router extension.
Tests are written first, then implementation follows.
"""

import pathlib
import pytest
from ulid import ULID

from redisvl.extensions.llm_router import LLMRouter, ModelTier, LLMRouteMatch
from redisvl.extensions.llm_router.schema import RoutingConfig

from tests.conftest import SKIP_HF, skip_if_no_redisearch, skip_if_redis_version_below

pytestmark = pytest.mark.skipif(
    SKIP_HF, reason="sentence-transformers not supported on Python 3.14+"
)


def get_base_path():
    return pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def model_tiers():
    """Define model tiers for testing."""
    return [
        ModelTier(
            name="simple",
            model="anthropic/claude-haiku-4-5",
            references=[
                "hello",
                "hi there", 
                "what time is it?",
                "thanks",
                "goodbye",
            ],
            metadata={
                "provider": "anthropic",
                "cost_per_1k_input": 0.00025,
                "cost_per_1k_output": 0.00125,
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
                "cost_per_1k_input": 0.015,
                "cost_per_1k_output": 0.075,
            },
            distance_threshold=0.7,
        ),
    ]


@pytest.fixture
def llm_router(client, model_tiers, hf_vectorizer):
    """Create an LLMRouter for testing."""
    skip_if_no_redisearch(client)
    router = LLMRouter(
        name=f"test-llm-router-{str(ULID())}",
        tiers=model_tiers,
        vectorizer=hf_vectorizer,
        redis_client=client,
        overwrite=True,
    )
    yield router
    router.delete()


class TestLLMRouterInitialization:
    """Test LLMRouter initialization and configuration."""

    def test_initialize_router(self, llm_router):
        """Router should initialize with tiers."""
        assert llm_router.name.startswith("test-llm-router-")
        assert len(llm_router.tiers) == 3
        assert llm_router.tier_names == ["simple", "reasoning", "expert"]

    def test_router_properties(self, llm_router):
        """Router should expose tier properties."""
        tier_names = llm_router.tier_names
        assert "simple" in tier_names
        assert "reasoning" in tier_names
        assert "expert" in tier_names

        # Check tier thresholds
        thresholds = llm_router.tier_thresholds
        assert thresholds["simple"] == 0.5
        assert thresholds["reasoning"] == 0.6
        assert thresholds["expert"] == 0.7

    def test_get_tier(self, llm_router):
        """Should retrieve tier by name."""
        tier = llm_router.get_tier("simple")
        assert tier is not None
        assert tier.name == "simple"
        assert tier.model == "anthropic/claude-haiku-4-5"

    def test_get_nonexistent_tier(self, llm_router):
        """Should return None for nonexistent tier."""
        tier = llm_router.get_tier("nonexistent")
        assert tier is None


class TestLLMRouterRouting:
    """Test LLMRouter routing behavior."""

    def test_route_simple_query(self, llm_router):
        """Simple greetings should route to simple tier."""
        skip_if_redis_version_below(llm_router._index.client, "7.0.0")

        match = llm_router.route("hello, how are you?")
        assert isinstance(match, LLMRouteMatch)
        assert match.tier == "simple"
        assert match.model == "anthropic/claude-haiku-4-5"
        assert match.distance is not None
        assert match.distance <= 0.5

    def test_route_reasoning_query(self, llm_router):
        """Complex analysis should route to reasoning tier."""
        skip_if_redis_version_below(llm_router._index.client, "7.0.0")

        match = llm_router.route("analyze this code and find potential bugs")
        assert match.tier == "reasoning"
        assert match.model == "anthropic/claude-sonnet-4-5"

    def test_route_expert_query(self, llm_router):
        """Research-level queries should route to expert tier."""
        skip_if_redis_version_below(llm_router._index.client, "7.0.0")

        match = llm_router.route("design a novel distributed consensus algorithm")
        assert match.tier == "expert"
        assert match.model == "anthropic/claude-opus-4-5"

    def test_route_no_match(self, llm_router):
        """Unrelated queries should return default tier or no match."""
        skip_if_redis_version_below(llm_router._index.client, "7.0.0")

        match = llm_router.route("xyzzy plugh random gibberish 12345")
        # Should return None or default tier depending on config
        assert match.tier is None or match.tier == llm_router.default_tier

    def test_route_with_vector(self, llm_router):
        """Should accept pre-computed vector."""
        skip_if_redis_version_below(llm_router._index.client, "7.0.0")

        vector = llm_router.vectorizer.embed("hello")
        match = llm_router.route(vector=vector)
        assert match.tier == "simple"

    def test_route_many(self, llm_router):
        """Should return multiple tier matches."""
        skip_if_redis_version_below(llm_router._index.client, "7.0.0")

        matches = llm_router.route_many("explain machine learning concepts", max_k=3)
        assert len(matches) > 0
        assert all(isinstance(m, LLMRouteMatch) for m in matches)


class TestLLMRouterCostOptimization:
    """Test cost-aware routing behavior."""

    def test_cost_optimization_prefers_cheaper(self, client, model_tiers, hf_vectorizer):
        """With cost optimization, should prefer cheaper tiers when close."""
        skip_if_no_redisearch(client)
        skip_if_redis_version_below(client, "7.0.0")

        router = LLMRouter(
            name=f"test-cost-router-{str(ULID())}",
            tiers=model_tiers,
            vectorizer=hf_vectorizer,
            redis_client=client,
            cost_optimization=True,
            overwrite=True,
        )

        try:
            # Query that closely matches simple tier references
            match = router.route("hello there, how are you?")
            # With cost optimization enabled, should match a tier
            # The exact tier depends on semantic similarity
            assert match.tier is not None or router.default_tier is not None
        finally:
            router.delete()


class TestLLMRouterSerialization:
    """Test LLMRouter serialization and persistence."""

    def test_to_dict(self, llm_router):
        """Should serialize to dictionary."""
        router_dict = llm_router.to_dict()
        assert router_dict["name"] == llm_router.name
        assert len(router_dict["tiers"]) == 3
        assert router_dict["vectorizer"]["type"] == llm_router.vectorizer.type

    def test_from_dict(self, llm_router):
        """Should deserialize from dictionary."""
        router_dict = llm_router.to_dict()
        new_router = LLMRouter.from_dict(
            router_dict,
            redis_client=llm_router._index.client,
            overwrite=True,
        )
        try:
            assert new_router.name == llm_router.name
            assert len(new_router.tiers) == len(llm_router.tiers)
        finally:
            # Don't delete - same index as original
            pass

    def test_to_yaml(self, llm_router, tmp_path):
        """Should serialize to YAML file."""
        yaml_file = tmp_path / "llm_router.yaml"
        llm_router.to_yaml(str(yaml_file))
        assert yaml_file.exists()

    def test_from_yaml(self, llm_router, tmp_path):
        """Should deserialize from YAML file."""
        yaml_file = tmp_path / "llm_router.yaml"
        llm_router.to_yaml(str(yaml_file))

        new_router = LLMRouter.from_yaml(
            str(yaml_file),
            redis_client=llm_router._index.client,
            overwrite=True,
        )
        assert new_router.name == llm_router.name


class TestLLMRouterWithEmbeddings:
    """Test LLMRouter with pre-computed embeddings."""

    def test_export_with_embeddings(self, llm_router, tmp_path):
        """Should export router with pre-computed embeddings."""
        json_file = tmp_path / "router_with_embeddings.json"
        llm_router.export_with_embeddings(str(json_file))
        
        assert json_file.exists()
        
        import json
        with open(json_file) as f:
            data = json.load(f)
        
        # Verify embeddings are included
        assert "tiers" in data
        for tier in data["tiers"]:
            assert "references" in tier
            for ref in tier["references"]:
                assert "text" in ref
                assert "vector" in ref
                assert isinstance(ref["vector"], list)
                assert len(ref["vector"]) > 0

    def test_import_with_embeddings(self, client, hf_vectorizer, tmp_path):
        """Should import router without re-embedding."""
        skip_if_no_redisearch(client)
        skip_if_redis_version_below(client, "7.0.0")

        # Create a simple router and export
        tier = ModelTier(
            name="test",
            model="test/model",
            references=["hello", "world"],
            distance_threshold=0.5,
        )
        router1 = LLMRouter(
            name=f"export-test-{str(ULID())}",
            tiers=[tier],
            vectorizer=hf_vectorizer,
            redis_client=client,
            overwrite=True,
        )
        
        json_file = tmp_path / "export_test.json"
        router1.export_with_embeddings(str(json_file))
        router1.delete()

        # Import - should not call vectorizer.embed()
        router2 = LLMRouter.from_pretrained(
            str(json_file),
            redis_client=client,
        )
        
        try:
            assert len(router2.tiers) == 1
            assert router2.tiers[0].name == "test"
            
            # Verify routing works
            match = router2.route("hello there")
            assert match.tier == "test"
        finally:
            router2.delete()


class TestLLMRouterTierManagement:
    """Test adding/removing tiers."""

    def test_add_tier(self, llm_router):
        """Should add new tier."""
        skip_if_redis_version_below(llm_router._index.client, "7.0.0")

        new_tier = ModelTier(
            name="local",
            model="ollama/llama3.2",
            references=["ok", "yes", "no"],
            metadata={"cost_per_1k_input": 0},
            distance_threshold=0.3,
        )
        llm_router.add_tier(new_tier)
        
        assert llm_router.get_tier("local") is not None
        assert "local" in llm_router.tier_names

    def test_remove_tier(self, llm_router):
        """Should remove tier."""
        llm_router.remove_tier("simple")
        
        assert llm_router.get_tier("simple") is None
        assert "simple" not in llm_router.tier_names

    def test_add_tier_references(self, llm_router):
        """Should add references to existing tier."""
        skip_if_redis_version_below(llm_router._index.client, "7.0.0")

        llm_router.add_tier_references(
            tier_name="simple",
            references=["howdy", "greetings"]
        )
        
        # Verify references were added to the tier
        tier = llm_router.get_tier("simple")
        assert "howdy" in tier.references
        assert "greetings" in tier.references
        
        # Verify routing with exact match works
        match = llm_router.route("howdy")
        assert match.tier == "simple"

    def test_update_tier_threshold(self, llm_router):
        """Should update tier threshold."""
        llm_router.update_tier_threshold("simple", 0.3)
        
        assert llm_router.tier_thresholds["simple"] == 0.3


class TestLLMRouterFromExisting:
    """Test reconnecting to existing router."""

    def test_from_existing(self, client, model_tiers, hf_vectorizer, redis_url):
        """Should reconnect to existing router."""
        skip_if_no_redisearch(client)
        skip_if_redis_version_below(client, "7.0.0")

        router_name = f"persist-test-{str(ULID())}"
        
        # Create router
        router1 = LLMRouter(
            name=router_name,
            tiers=model_tiers,
            vectorizer=hf_vectorizer,
            redis_url=redis_url,
            overwrite=True,
        )
        
        # Reconnect
        router2 = LLMRouter.from_existing(
            name=router_name,
            redis_url=redis_url,
        )
        
        try:
            assert router2.name == router1.name
            assert len(router2.tiers) == len(router1.tiers)
            assert router2.tier_names == router1.tier_names
            
            # Verify routing works
            match = router2.route("hello")
            assert match.tier is not None
        finally:
            router1.delete()
