"""SVS-VAMANA compression configuration utilities."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SVSConfig(BaseModel):
    """SVS-VAMANA configuration model.

    Attributes:
        algorithm: Always "svs-vamana"
        datatype: Vector datatype (float16, float32)
        compression: Compression type (LVQ4, LeanVec4x8, etc.)
        reduce: Reduced dimensionality (only for LeanVec)
        graph_max_degree: Max edges per node
        construction_window_size: Build-time candidates
        search_window_size: Query-time candidates
    """

    algorithm: Literal["svs-vamana"] = "svs-vamana"
    datatype: Optional[str] = None
    compression: Optional[str] = None
    reduce: Optional[int] = Field(
        default=None, description="Reduced dimensionality (only for LeanVec)"
    )
    graph_max_degree: Optional[int] = None
    construction_window_size: Optional[int] = None
    search_window_size: Optional[int] = None


class CompressionAdvisor:
    """Helper to recommend compression settings based on vector characteristics.

    This class provides utilities to:
    - Recommend optimal SVS-VAMANA configurations based on vector dimensions and priorities
    - Estimate memory savings from compression and dimensionality reduction

    Examples:
        >>> # Get recommendations for high-dimensional vectors
        >>> config = CompressionAdvisor.recommend(dims=1536, priority="balanced")
        >>> config.compression
        'LeanVec4x8'
        >>> config.reduce
        768

        >>> # Estimate memory savings
        >>> savings = CompressionAdvisor.estimate_memory_savings(
        ...     compression="LeanVec4x8",
        ...     dims=1536,
        ...     reduce=768
        ... )
        >>> savings
        81.2
    """

    # Dimension thresholds
    HIGH_DIM_THRESHOLD = 1024

    # Compression bit rates (bits per dimension)
    COMPRESSION_BITS = {
        "LVQ4": 4,
        "LVQ4x4": 8,
        "LVQ4x8": 12,
        "LVQ8": 8,
        "LeanVec4x8": 12,
        "LeanVec8x8": 16,
    }

    @staticmethod
    def recommend(
        dims: int,
        priority: Literal["speed", "memory", "balanced"] = "balanced",
        datatype: Optional[str] = None,
    ) -> SVSConfig:
        """Recommend compression settings based on dimensions and priorities.

        Args:
            dims: Vector dimensionality (must be > 0)
            priority: Optimization priority:
                - "memory": Maximize memory savings
                - "speed": Optimize for query speed
                - "balanced": Balance between memory and speed
            datatype: Override datatype (default: float16 for high-dim, float32 for low-dim)

        Returns:
            dict: Complete SVS-VAMANA configuration including:
                - algorithm: "svs-vamana"
                - datatype: Recommended datatype
                - compression: Compression type
                - reduce: Dimensionality reduction (for LeanVec only)
                - graph_max_degree: Graph connectivity
                - construction_window_size: Build-time candidates
                - search_window_size: Query-time candidates

        Raises:
            ValueError: If dims <= 0

        Examples:
            >>> # High-dimensional embeddings (e.g., OpenAI ada-002)
            >>> config = CompressionAdvisor.recommend(dims=1536, priority="memory")
            >>> config.compression
            'LeanVec4x8'
            >>> config.reduce
            768

            >>> # Lower-dimensional embeddings
            >>> config = CompressionAdvisor.recommend(dims=384, priority="speed")
            >>> config.compression
            'LVQ4x8'
        """
        if dims <= 0:
            raise ValueError(f"dims must be positive, got {dims}")

        # High-dimensional vectors (>= 1024) - use LeanVec
        if dims >= CompressionAdvisor.HIGH_DIM_THRESHOLD:
            base_datatype = datatype or "float16"

            if priority == "memory":
                return SVSConfig(
                    algorithm="svs-vamana",
                    datatype=base_datatype,
                    graph_max_degree=64,
                    construction_window_size=300,
                    compression="LeanVec4x8",
                    reduce=dims // 2,
                    search_window_size=20,
                )
            elif priority == "speed":
                return SVSConfig(
                    algorithm="svs-vamana",
                    datatype=base_datatype,
                    graph_max_degree=64,
                    construction_window_size=300,
                    compression="LeanVec4x8",
                    reduce=max(256, dims // 4),
                    search_window_size=40,
                )
            else:  # balanced
                return SVSConfig(
                    algorithm="svs-vamana",
                    datatype=base_datatype,
                    graph_max_degree=64,
                    construction_window_size=300,
                    compression="LeanVec4x8",
                    reduce=dims // 2,
                    search_window_size=30,
                )

        # Lower-dimensional vectors - use LVQ
        else:
            base_datatype = datatype or "float32"

            if priority == "memory":
                return SVSConfig(
                    algorithm="svs-vamana",
                    datatype=base_datatype,
                    graph_max_degree=40,
                    construction_window_size=250,
                    search_window_size=20,
                    compression="LVQ4",
                )
            elif priority == "speed":
                return SVSConfig(
                    algorithm="svs-vamana",
                    datatype=base_datatype,
                    graph_max_degree=40,
                    construction_window_size=250,
                    search_window_size=20,
                    compression="LVQ4x8",
                )
            else:  # balanced
                return SVSConfig(
                    algorithm="svs-vamana",
                    datatype=base_datatype,
                    graph_max_degree=40,
                    construction_window_size=250,
                    search_window_size=20,
                    compression="LVQ4x4",
                )

    @staticmethod
    def estimate_memory_savings(
        compression: str, dims: int, reduce: Optional[int] = None
    ) -> float:
        """Estimate memory savings percentage from compression.

        Calculates the percentage of memory saved compared to uncompressed float32 vectors.

        Args:
            compression: Compression type (e.g., "LVQ4", "LeanVec4x8")
            dims: Original vector dimensionality
            reduce: Reduced dimensionality (for LeanVec compression)

        Returns:
            float: Memory savings percentage (0-100)

        Examples:
            >>> # LeanVec with dimensionality reduction
            >>> CompressionAdvisor.estimate_memory_savings(
            ...     compression="LeanVec4x8",
            ...     dims=1536,
            ...     reduce=768
            ... )
            81.2

            >>> # LVQ without dimensionality reduction
            >>> CompressionAdvisor.estimate_memory_savings(
            ...     compression="LVQ4",
            ...     dims=384
            ... )
            87.5
        """
        # Base bits per dimension (float32)
        base_bits = 32

        # Compressed bits per dimension
        compression_bits = CompressionAdvisor.COMPRESSION_BITS.get(
            compression, base_bits
        )

        # Account for dimensionality reduction
        effective_dims = reduce if reduce else dims

        # Calculate savings
        original_size = dims * base_bits
        compressed_size = effective_dims * compression_bits
        savings = (1 - compressed_size / original_size) * 100

        return round(savings, 1)
