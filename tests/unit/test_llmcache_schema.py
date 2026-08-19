import json

import pytest
from pydantic import ValidationError

from redisvl.extensions.cache.llm import CacheEntry, CacheHit, SemanticCacheIndexSchema
from redisvl.extensions.constants import CACHE_VECTOR_FIELD_NAME
from redisvl.redis.utils import array_to_buffer, hashify


def test_valid_cache_entry_creation():
    entry = CacheEntry(
        prompt="What is AI?",
        response="AI is artificial intelligence.",
        prompt_vector=[0.1, 0.2, 0.3],
    )
    assert entry.entry_id == hashify("What is AI?")
    assert entry.prompt == "What is AI?"
    assert entry.response == "AI is artificial intelligence."
    assert entry.prompt_vector == [0.1, 0.2, 0.3]


def test_cache_entry_with_given_entry_id():
    entry = CacheEntry(
        entry_id="custom_id",
        prompt="What is AI?",
        response="AI is artificial intelligence.",
        prompt_vector=[0.1, 0.2, 0.3],
    )
    assert entry.entry_id == "custom_id"


def test_cache_entry_with_invalid_metadata():
    with pytest.raises(ValidationError):
        CacheEntry(
            prompt="What is AI?",
            response="AI is artificial intelligence.",
            prompt_vector=[0.1, 0.2, 0.3],
            metadata="invalid_metadata",
        )


def test_cache_entry_to_dict():
    entry = CacheEntry(
        prompt="What is AI?",
        response="AI is artificial intelligence.",
        prompt_vector=[0.1, 0.2, 0.3],
        metadata={"author": "John"},
        filters={"category": "technology"},
    )
    result = entry.to_dict(dtype="float32")
    assert result["entry_id"] == hashify("What is AI?", {"category": "technology"})
    assert result["metadata"] == json.dumps({"author": "John"})
    assert result["prompt_vector"] == array_to_buffer([0.1, 0.2, 0.3], "float32")
    assert result["category"] == "technology"
    assert "filters" not in result


def test_valid_cache_hit_creation():
    hit = CacheHit(
        entry_id="entry_1",
        prompt="What is AI?",
        response="AI is artificial intelligence.",
        vector_distance=0.1,
        inserted_at=1625819123.123,
        updated_at=1625819123.123,
    )
    assert hit.entry_id == "entry_1"
    assert hit.prompt == "What is AI?"
    assert hit.response == "AI is artificial intelligence."
    assert hit.vector_distance == 0.1
    assert hit.inserted_at == hit.updated_at == 1625819123.123


def test_cache_hit_with_serialized_metadata():
    hit = CacheHit(
        entry_id="entry_1",
        prompt="What is AI?",
        response="AI is artificial intelligence.",
        vector_distance=0.1,
        inserted_at=1625819123.123,
        updated_at=1625819123.123,
        metadata=json.dumps({"author": "John"}),
    )
    assert hit.metadata == {"author": "John"}


def test_cache_hit_to_dict():
    hit = CacheHit(
        entry_id="entry_1",
        prompt="What is AI?",
        response="AI is artificial intelligence.",
        vector_distance=0.1,
        inserted_at=1625819123.123,
        updated_at=1625819123.123,
        filters={"category": "technology"},
    )
    result = hit.to_dict()
    assert result["entry_id"] == "entry_1"
    assert result["prompt"] == "What is AI?"
    assert result["response"] == "AI is artificial intelligence."
    assert result["vector_distance"] == 0.1
    assert result["category"] == "technology"
    assert "filters" not in result


def test_cache_entry_with_empty_optional_fields():
    entry = CacheEntry(
        prompt="What is AI?",
        response="AI is artificial intelligence.",
        prompt_vector=[0.1, 0.2, 0.3],
    )
    result = entry.to_dict(dtype="float32")
    assert "metadata" not in result
    assert "filters" not in result


def test_cache_hit_with_empty_optional_fields():
    hit = CacheHit(
        entry_id="entry_1",
        prompt="What is AI?",
        response="AI is artificial intelligence.",
        vector_distance=0.1,
        inserted_at=1625819123.123,
        updated_at=1625819123.123,
    )
    result = hit.to_dict()
    assert "metadata" not in result
    assert "filters" not in result


def test_semantic_cache_schema_defaults_to_flat_vector_index():
    schema = SemanticCacheIndexSchema.from_params(
        "llmcache", "llmcache", vector_dims=768, dtype="float32"
    )

    vector_field = schema.fields[CACHE_VECTOR_FIELD_NAME]

    assert vector_field.attrs.algorithm.value == "FLAT"
    assert vector_field.attrs.dims == 768
    assert vector_field.attrs.datatype.value == "FLOAT32"
    assert vector_field.attrs.distance_metric.value == "COSINE"


def test_semantic_cache_schema_accepts_hnsw_vector_index_config():
    schema = SemanticCacheIndexSchema.from_params(
        "llmcache",
        "llmcache",
        vector_dims=768,
        dtype="float32",
        vector_index_config={
            "algorithm": "hnsw",
            "m": 32,
            "ef_construction": 300,
            "ef_runtime": 20,
        },
    )

    vector_field = schema.fields[CACHE_VECTOR_FIELD_NAME]

    assert vector_field.attrs.algorithm.value == "HNSW"
    assert vector_field.attrs.dims == 768
    assert vector_field.attrs.datatype.value == "FLOAT32"
    assert vector_field.attrs.distance_metric.value == "COSINE"
    assert vector_field.attrs.m == 32
    assert vector_field.attrs.ef_construction == 300
    assert vector_field.attrs.ef_runtime == 20


def test_semantic_cache_schema_rejects_unknown_vector_index_algorithm():
    with pytest.raises(ValueError, match="Unknown vector field algorithm"):
        SemanticCacheIndexSchema.from_params(
            "llmcache",
            "llmcache",
            vector_dims=768,
            dtype="float32",
            vector_index_config={"algorithm": "unknown"},
        )


def test_semantic_cache_schema_rejects_vectorizer_derived_config_overrides():
    with pytest.raises(ValueError, match="dims"):
        SemanticCacheIndexSchema.from_params(
            "llmcache",
            "llmcache",
            vector_dims=768,
            dtype="float32",
            vector_index_config={"algorithm": "hnsw", "dims": 1536},
        )
