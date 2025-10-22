"""Unit tests for CompressionAdvisor utility."""

import pytest

from redisvl.utils.compression import CompressionAdvisor, SVSConfig


class TestCompressionAdvisorRecommend:
    """Tests for CompressionAdvisor.recommend() method."""

    def test_recommend_high_dim_memory_priority(self):
        """Test memory-optimized config for high-dimensional vectors."""
        config = CompressionAdvisor.recommend(dims=1536, priority="memory")

        assert config.algorithm == "svs-vamana"
        assert config.datatype == "float16"
        assert config.compression == "LeanVec4x8"
        assert config.reduce == 768  # dims // 2
        assert config.graph_max_degree == 64
        assert config.construction_window_size == 300
        assert config.search_window_size == 20

    def test_recommend_high_dim_speed_priority(self):
        """Test speed-optimized config for high-dimensional vectors."""
        config = CompressionAdvisor.recommend(dims=1536, priority="speed")

        assert config.algorithm == "svs-vamana"
        assert config.datatype == "float16"
        assert config.compression == "LeanVec4x8"
        assert config.reduce == 384  # dims // 4
        assert config.graph_max_degree == 64
        assert config.construction_window_size == 300
        assert config.search_window_size == 40

    def test_recommend_high_dim_balanced_priority(self):
        """Test balanced config for high-dimensional vectors."""
        config = CompressionAdvisor.recommend(dims=1536, priority="balanced")

        assert config.algorithm == "svs-vamana"
        assert config.datatype == "float16"
        assert config.compression == "LeanVec4x8"
        assert config.reduce == 768  # dims // 2
        assert config.graph_max_degree == 64
        assert config.construction_window_size == 300
        assert config.search_window_size == 30

    def test_recommend_high_dim_default_priority(self):
        """Test default priority (balanced) for high-dimensional vectors."""
        config = CompressionAdvisor.recommend(dims=2048)

        assert config.compression == "LeanVec4x8"
        assert config.reduce == 1024
        assert config.search_window_size == 30

    def test_recommend_low_dim_memory_priority(self):
        """Test memory-optimized config for low-dimensional vectors."""
        config = CompressionAdvisor.recommend(dims=384, priority="memory")

        assert config.algorithm == "svs-vamana"
        assert config.datatype == "float32"
        assert config.compression == "LVQ4"
        assert config.reduce is None  # LVQ doesn't use reduce
        assert config.graph_max_degree == 40
        assert config.construction_window_size == 250
        assert config.search_window_size == 20

    def test_recommend_low_dim_speed_priority(self):
        """Test speed-optimized config for low-dimensional vectors."""
        config = CompressionAdvisor.recommend(dims=384, priority="speed")

        assert config.algorithm == "svs-vamana"
        assert config.datatype == "float32"
        assert config.compression == "LVQ4x8"
        assert config.reduce is None
        assert config.graph_max_degree == 40
        assert config.construction_window_size == 250
        assert config.search_window_size == 20

    def test_recommend_low_dim_balanced_priority(self):
        """Test balanced config for low-dimensional vectors."""
        config = CompressionAdvisor.recommend(dims=768, priority="balanced")

        assert config.algorithm == "svs-vamana"
        assert config.datatype == "float32"
        assert config.compression == "LVQ4x4"
        assert config.reduce is None
        assert config.graph_max_degree == 40
        assert config.construction_window_size == 250
        assert config.search_window_size == 20

    def test_recommend_threshold_boundary_low(self):
        """Test recommendation at threshold boundary (1023 dims)."""
        config = CompressionAdvisor.recommend(dims=1023)

        # Should use LVQ (below threshold)
        assert config.compression in ["LVQ4", "LVQ4x4", "LVQ4x8"]
        assert config.datatype == "float32"
        assert config.reduce is None

    def test_recommend_threshold_boundary_high(self):
        """Test recommendation at threshold boundary (1024 dims)."""
        config = CompressionAdvisor.recommend(dims=1024)

        # Should use LeanVec (at threshold)
        assert config.compression == "LeanVec4x8"
        assert config.datatype == "float16"
        assert config.reduce is not None

    def test_recommend_custom_datatype(self):
        """Test custom datatype override."""
        config = CompressionAdvisor.recommend(dims=1536, datatype="float32")

        assert config.datatype == "float32"

    def test_recommend_speed_reduce_minimum(self):
        """Test that speed priority respects minimum reduce value."""
        config = CompressionAdvisor.recommend(dims=1024, priority="speed")

        # dims // 4 = 256, max(256, 256) = 256
        assert config.reduce == 256

        config = CompressionAdvisor.recommend(dims=512, priority="speed")
        # Below threshold, should use LVQ
        assert config.reduce is None

    def test_recommend_invalid_dims_zero(self):
        """Test that zero dims raises ValueError."""
        with pytest.raises(ValueError, match="dims must be positive"):
            CompressionAdvisor.recommend(dims=0)

    def test_recommend_invalid_dims_negative(self):
        """Test that negative dims raises ValueError."""
        with pytest.raises(ValueError, match="dims must be positive"):
            CompressionAdvisor.recommend(dims=-100)


class TestCompressionAdvisorEstimateMemorySavings:
    """Tests for CompressionAdvisor.estimate_memory_savings() method."""

    def test_estimate_lvq4_no_reduce(self):
        """Test memory savings for LVQ4 without dimensionality reduction."""
        savings = CompressionAdvisor.estimate_memory_savings(
            compression="LVQ4", dims=384
        )

        # Original: 384 * 32 = 12,288 bits
        # Compressed: 384 * 4 = 1,536 bits
        # Savings: (1 - 1536/12288) * 100 = 87.5%
        assert savings == 87.5

    def test_estimate_lvq4x4_no_reduce(self):
        """Test memory savings for LVQ4x4 without dimensionality reduction."""
        savings = CompressionAdvisor.estimate_memory_savings(
            compression="LVQ4x4", dims=768
        )

        # Original: 768 * 32 = 24,576 bits
        # Compressed: 768 * 8 = 6,144 bits
        # Savings: (1 - 6144/24576) * 100 = 75.0%
        assert savings == 75.0

    def test_estimate_leanvec4x8_with_reduce(self):
        """Test memory savings for LeanVec4x8 with dimensionality reduction."""
        savings = CompressionAdvisor.estimate_memory_savings(
            compression="LeanVec4x8", dims=1536, reduce=768
        )

        # Original: 1536 * 32 = 49,152 bits
        # Compressed: 768 * 12 = 9,216 bits
        # Savings: (1 - 9216/49152) * 100 = 81.25% -> 81.2% (rounded)
        assert savings == 81.2

    def test_estimate_leanvec4x8_no_reduce(self):
        """Test memory savings for LeanVec4x8 without dimensionality reduction."""
        savings = CompressionAdvisor.estimate_memory_savings(
            compression="LeanVec4x8", dims=1536
        )

        # Original: 1536 * 32 = 49,152 bits
        # Compressed: 1536 * 12 = 18,432 bits
        # Savings: (1 - 18432/49152) * 100 = 62.5%
        assert savings == 62.5

    def test_estimate_leanvec8x8_with_reduce(self):
        """Test memory savings for LeanVec8x8 with dimensionality reduction."""
        savings = CompressionAdvisor.estimate_memory_savings(
            compression="LeanVec8x8", dims=2048, reduce=1024
        )

        # Original: 2048 * 32 = 65,536 bits
        # Compressed: 1024 * 16 = 16,384 bits
        # Savings: (1 - 16384/65536) * 100 = 75.0%
        assert savings == 75.0

    def test_estimate_unknown_compression(self):
        """Test that unknown compression type defaults to no savings."""
        savings = CompressionAdvisor.estimate_memory_savings(
            compression="UNKNOWN", dims=512
        )

        # Should default to base_bits (32), so no savings
        # Original: 512 * 32 = 16,384 bits
        # Compressed: 512 * 32 = 16,384 bits
        # Savings: 0%
        assert savings == 0.0

    def test_estimate_rounding(self):
        """Test that savings are rounded to 1 decimal place."""
        savings = CompressionAdvisor.estimate_memory_savings(
            compression="LVQ4", dims=333
        )

        # Original: 333 * 32 = 10,656 bits
        # Compressed: 333 * 4 = 1,332 bits
        # Savings: (1 - 1332/10656) * 100 = 87.5%
        assert savings == 87.5
        assert isinstance(savings, float)


class TestSVSConfigModel:
    """Tests for SVSConfig Pydantic model structure."""

    def test_svs_config_structure(self):
        """Test that SVSConfig can be constructed with all fields."""
        config = SVSConfig(
            algorithm="svs-vamana",
            datatype="float16",
            compression="LeanVec4x8",
            reduce=768,
            graph_max_degree=64,
            construction_window_size=300,
            search_window_size=30,
        )

        assert config.algorithm == "svs-vamana"
        assert config.reduce == 768

    def test_svs_config_without_reduce(self):
        """Test that SVSConfig can be constructed without reduce field."""
        config = SVSConfig(
            algorithm="svs-vamana",
            datatype="float32",
            compression="LVQ4",
            graph_max_degree=40,
            construction_window_size=250,
            search_window_size=20,
        )

        assert config.reduce is None
        assert config.compression == "LVQ4"
