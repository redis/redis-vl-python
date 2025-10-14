"""
Integration tests for SVS-VAMANA vector indexing.

These tests require:
- Redis >= 8.2.0
- RediSearch >= 2.8.10
- Environment variable: REDISVL_TEST_SVS=1

Run with:
    REDISVL_TEST_SVS=1 pytest tests/integration/test_svs_integration.py

To run with Redis 8.2+ in Docker:
    docker run -d -p 6379:6379 redis/redis-stack-server:8.2.2-v0
    REDISVL_TEST_SVS=1 pytest tests/integration/test_svs_integration.py

Or set the Redis image for the test suite:
    REDIS_IMAGE=redis/redis-stack-server:8.2.2-v0 REDISVL_TEST_SVS=1 pytest tests/integration/test_svs_integration.py
"""

import os

import numpy as np
import pytest

from redisvl.exceptions import RedisModuleVersionError
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.redis.connection import supports_svs
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema
from redisvl.utils import CompressionAdvisor

# Skip all tests in this module if REDISVL_TEST_SVS is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("REDISVL_TEST_SVS"),
    reason="SVS tests require REDISVL_TEST_SVS=1 and Redis 8.2+",
)


@pytest.fixture
def svs_schema_lvq(worker_id):
    """Create SVS-VAMANA schema with LVQ compression."""
    return IndexSchema.from_dict(
        {
            "index": {
                "name": f"svs_lvq_{worker_id}",
                "prefix": f"svs_lvq_{worker_id}",
            },
            "fields": [
                {"name": "id", "type": "tag"},
                {"name": "content", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 384,
                        "algorithm": "svs-vamana",
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "compression": "LVQ4x4",
                        "graph_max_degree": 40,
                        "construction_window_size": 250,
                        "search_window_size": 20,
                    },
                },
            ],
        }
    )


@pytest.fixture
def svs_schema_leanvec(worker_id):
    """Create SVS-VAMANA schema with LeanVec compression and dimensionality reduction."""
    return IndexSchema.from_dict(
        {
            "index": {
                "name": f"svs_leanvec_{worker_id}",
                "prefix": f"svs_leanvec_{worker_id}",
            },
            "fields": [
                {"name": "id", "type": "tag"},
                {"name": "content", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 1536,
                        "algorithm": "svs-vamana",
                        "datatype": "float16",
                        "distance_metric": "cosine",
                        "compression": "LeanVec4x8",
                        "reduce": 768,
                        "graph_max_degree": 64,
                        "construction_window_size": 300,
                        "search_window_size": 30,
                    },
                },
            ],
        }
    )


@pytest.fixture
def svs_index_lvq(svs_schema_lvq, client):
    """Create SVS-VAMANA index with LVQ compression."""
    index = SearchIndex(schema=svs_schema_lvq, redis_client=client)
    index.create(overwrite=True)
    yield index
    index.delete(drop=True)


@pytest.fixture
def svs_index_leanvec(svs_schema_leanvec, client):
    """Create SVS-VAMANA index with LeanVec compression."""
    index = SearchIndex(schema=svs_schema_leanvec, redis_client=client)
    index.create(overwrite=True)
    yield index
    index.delete(drop=True)


def generate_test_vectors(dims, count=100, dtype="float32"):
    """Generate random test vectors."""
    vectors = []
    for i in range(count):
        vector = np.random.random(dims).astype(dtype)
        vectors.append(
            {
                "id": f"doc_{i}",
                "content": f"This is test document {i}",
                "embedding": array_to_buffer(vector, dtype=dtype),
            }
        )
    return vectors


class TestSVSCapabilityDetection:
    """Test SVS-VAMANA capability detection."""

    def test_check_svs_capabilities(self, client):
        """Test that SVS-VAMANA is supported on the test Redis instance."""
        # These tests require Redis 8.2+ with RediSearch 2.8.10+
        assert supports_svs(client) is True, (
            "SVS-VAMANA not supported. "
            "Requires Redis >= 8.2.0 with RediSearch >= 2.8.10"
        )


class TestSVSIndexCreation:
    """Test creating SVS-VAMANA indices with various configurations."""

    def test_create_svs_index_lvq(self, svs_index_lvq):
        """Test creating SVS-VAMANA index with LVQ compression."""
        assert svs_index_lvq.exists()

        # Verify index info
        info = svs_index_lvq.info()
        assert info["num_docs"] == 0

    def test_create_svs_index_leanvec(self, svs_index_leanvec):
        """Test creating SVS-VAMANA index with LeanVec compression."""
        assert svs_index_leanvec.exists()

        # Verify index info
        info = svs_index_leanvec.info()
        assert info["num_docs"] == 0

    def test_create_svs_with_compression_advisor(self, client, worker_id):
        """Test creating SVS-VAMANA index using CompressionAdvisor."""
        dims = 768
        config = CompressionAdvisor.recommend(dims=dims, priority="balanced")

        schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": f"svs_advisor_{worker_id}",
                    "prefix": f"svs_advisor_{worker_id}",
                },
                "fields": [
                    {"name": "id", "type": "tag"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": dims,
                            **config,
                            "distance_metric": "cosine",
                        },
                    },
                ],
            }
        )

        index = SearchIndex(schema=schema, redis_client=client)
        index.create(overwrite=True)

        try:
            assert index.exists()
            info = index.info()
            assert info["num_docs"] == 0
        finally:
            index.delete(drop=True)


class TestSVSDataIngestion:
    """Test loading data into SVS-VAMANA indices."""

    def test_load_data_lvq(self, svs_index_lvq):
        """Test loading data into SVS-VAMANA index with LVQ compression."""
        vectors = generate_test_vectors(dims=384, count=50, dtype="float32")
        svs_index_lvq.load(vectors)

        # Verify data was loaded
        info = svs_index_lvq.info()
        assert info["num_docs"] == 50

    def test_load_data_leanvec(self, svs_index_leanvec):
        """Test loading data into SVS-VAMANA index with LeanVec compression."""
        vectors = generate_test_vectors(dims=1536, count=50, dtype="float32")
        svs_index_leanvec.load(vectors)

        # Verify data was loaded
        info = svs_index_leanvec.info()
        assert info["num_docs"] == 50

    def test_load_large_batch(self, svs_index_lvq):
        """Test loading larger batch of data."""
        vectors = generate_test_vectors(dims=384, count=200, dtype="float32")
        svs_index_lvq.load(vectors)

        # Verify data was loaded
        info = svs_index_lvq.info()
        assert info["num_docs"] == 200


class TestSVSQuerying:
    """Test querying SVS-VAMANA indices."""

    def test_vector_query_lvq(self, svs_index_lvq):
        """Test vector similarity search on SVS-VAMANA index with LVQ."""
        # Load test data
        vectors = generate_test_vectors(dims=384, count=100, dtype="float32")
        svs_index_lvq.load(vectors)

        # Create query vector
        query_vector = np.random.random(384).astype(np.float32)

        # Execute query
        query = VectorQuery(
            vector=query_vector,
            vector_field_name="embedding",
            return_fields=["id", "content"],
            num_results=10,
        )

        results = svs_index_lvq.query(query)

        # Verify results
        assert len(results) <= 10
        assert all("id" in result for result in results)
        assert all("content" in result for result in results)

    def test_vector_query_leanvec(self, svs_index_leanvec):
        """Test vector similarity search on SVS-VAMANA index with LeanVec."""
        # Load test data
        vectors = generate_test_vectors(dims=1536, count=100, dtype="float32")
        svs_index_leanvec.load(vectors)

        # Create query vector
        query_vector = np.random.random(1536).astype(np.float32)

        # Execute query
        query = VectorQuery(
            vector=query_vector,
            vector_field_name="embedding",
            return_fields=["id", "content"],
            num_results=5,
        )

        results = svs_index_leanvec.query(query)

        # Verify results
        assert len(results) <= 5
        assert all("id" in result for result in results)

    def test_query_with_filters(self, svs_index_lvq):
        """Test vector query with filters on SVS-VAMANA index."""
        # Load test data with specific IDs
        vectors = []
        for i in range(50):
            vector = np.random.random(384).astype(np.float32)
            vectors.append(
                {
                    "id": f"category_a_{i}" if i < 25 else f"category_b_{i}",
                    "content": f"Document {i}",
                    "embedding": array_to_buffer(vector, dtype="float32"),
                }
            )
        svs_index_lvq.load(vectors)

        # Query with filter
        query_vector = np.random.random(384).astype(np.float32)
        query = VectorQuery(
            vector=query_vector,
            vector_field_name="embedding",
            return_fields=["id", "content"],
            num_results=10,
            filter_expression="@id:{category_a*}",
        )

        results = svs_index_lvq.query(query)

        # Verify all results match filter
        assert len(results) <= 10
        assert all(result["id"].startswith("category_a") for result in results)


class TestSVSFromExisting:
    """Test loading existing SVS-VAMANA indices."""

    def test_from_existing_lvq(self, svs_index_lvq, client):
        """Test loading existing SVS-VAMANA index with LVQ compression."""
        # Load some data
        vectors = generate_test_vectors(dims=384, count=20, dtype="float32")
        svs_index_lvq.load(vectors)

        # Load the index from existing
        loaded_index = SearchIndex.from_existing(
            svs_index_lvq.name, redis_client=client
        )

        # Verify the loaded index
        assert loaded_index.exists()
        assert loaded_index.name == svs_index_lvq.name

        # Verify schema was loaded correctly
        embedding_field = loaded_index.schema.fields["embedding"]
        assert embedding_field.attrs.algorithm.value == "SVS-VAMANA"
        assert embedding_field.attrs.compression.value == "LVQ4x4"
        assert embedding_field.attrs.dims == 384

        # Verify data is accessible
        info = loaded_index.info()
        assert info["num_docs"] == 20

    def test_from_existing_leanvec(self, svs_index_leanvec, client):
        """Test loading existing SVS-VAMANA index with LeanVec compression."""
        # Load some data
        vectors = generate_test_vectors(dims=1536, count=20, dtype="float32")
        svs_index_leanvec.load(vectors)

        # Load the index from existing
        loaded_index = SearchIndex.from_existing(
            svs_index_leanvec.name, redis_client=client
        )

        # Verify the loaded index
        assert loaded_index.exists()
        assert loaded_index.name == svs_index_leanvec.name

        # Verify schema was loaded correctly
        embedding_field = loaded_index.schema.fields["embedding"]
        assert embedding_field.attrs.algorithm.value == "SVS-VAMANA"
        assert embedding_field.attrs.compression.value == "LeanVec4x8"
        assert embedding_field.attrs.dims == 1536
        assert embedding_field.attrs.reduce == 768

        # Verify data is accessible
        info = loaded_index.info()
        assert info["num_docs"] == 20


class TestSVSCompressionTypes:
    """Test different compression types for SVS-VAMANA."""

    @pytest.mark.parametrize(
        "compression,dims,dtype",
        [
            ("LVQ4", 384, "float32"),
            ("LVQ4x4", 384, "float32"),
            ("LVQ4x8", 384, "float32"),
            ("LVQ8", 384, "float32"),
        ],
    )
    def test_lvq_compression_types(self, client, worker_id, compression, dims, dtype):
        """Test various LVQ compression types."""
        schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": f"svs_{compression.lower()}_{worker_id}",
                    "prefix": f"svs_{compression.lower()}_{worker_id}",
                },
                "fields": [
                    {"name": "id", "type": "tag"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": dims,
                            "algorithm": "svs-vamana",
                            "datatype": dtype,
                            "distance_metric": "cosine",
                            "compression": compression,
                        },
                    },
                ],
            }
        )

        index = SearchIndex(schema=schema, redis_client=client)
        index.create(overwrite=True)

        try:
            # Load data
            vectors = generate_test_vectors(dims=dims, count=50, dtype=dtype)
            index.load(vectors)

            # Verify
            assert index.exists()
            info = index.info()
            assert info["num_docs"] == 50

            # Query
            query_vector = np.random.random(dims).astype(dtype)
            query = VectorQuery(
                vector=query_vector,
                vector_field_name="embedding",
                return_fields=["id"],
                num_results=5,
            )
            results = index.query(query)
            assert len(results) <= 5
        finally:
            index.delete(drop=True)

    @pytest.mark.parametrize(
        "compression,dims,reduce,dtype",
        [
            ("LeanVec4x8", 1024, 512, "float16"),
            ("LeanVec4x8", 1536, 768, "float16"),
            ("LeanVec8x8", 1536, 768, "float16"),
        ],
    )
    def test_leanvec_compression_types(
        self, client, worker_id, compression, dims, reduce, dtype
    ):
        """Test various LeanVec compression types with dimensionality reduction."""
        schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": f"svs_{compression.lower()}_{reduce}_{worker_id}",
                    "prefix": f"svs_{compression.lower()}_{reduce}_{worker_id}",
                },
                "fields": [
                    {"name": "id", "type": "tag"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": dims,
                            "algorithm": "svs-vamana",
                            "datatype": dtype,
                            "distance_metric": "cosine",
                            "compression": compression,
                            "reduce": reduce,
                        },
                    },
                ],
            }
        )

        index = SearchIndex(schema=schema, redis_client=client)
        index.create(overwrite=True)

        try:
            # Load data
            vectors = generate_test_vectors(dims=dims, count=50, dtype="float32")
            index.load(vectors)

            # Verify
            assert index.exists()
            info = index.info()
            assert info["num_docs"] == 50

            # Query
            query_vector = np.random.random(dims).astype(np.float32)
            query = VectorQuery(
                vector=query_vector,
                vector_field_name="embedding",
                return_fields=["id"],
                num_results=5,
            )
            results = index.query(query)
            assert len(results) <= 5
        finally:
            index.delete(drop=True)
