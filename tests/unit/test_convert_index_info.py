"""Unit tests for convert_index_info_to_schema function."""

import pytest

from redisvl.redis.connection import convert_index_info_to_schema
from redisvl.schema import IndexSchema


def svs_info(*tail):
    """FT.INFO reply for an SVS-VAMANA field with the given trailing attributes."""
    return {
        "index_name": "svs",
        "index_definition": ["key_type", "HASH", "prefixes", ["svs"]],
        "attributes": [
            [
                "identifier",
                "emb",
                "attribute",
                "emb",
                "type",
                "VECTOR",
                "algorithm",
                "SVS-VAMANA",
                "data_type",
                "FLOAT32",
                "dim",
                4,
                "distance_metric",
                "COSINE",
                "graph_max_degree",
                32,
                "construction_window_size",
                200,
                *tail,
            ]
        ],
    }


@pytest.mark.parametrize(
    "tail, compression, threshold",
    [
        (["compression", "NO_COMPRESSION"], None, None),
        (["compression", "GlobalSQ8", "training_threshold", 2048], "LVQ4x4", 2048),
        (["compression", "GlobalSQ8", "training_threshold", 10240], "LVQ4x4", None),
    ],
)
def test_convert_index_info_svs_compression(tail, compression, threshold):
    """Redis 8.2+ reply shapes; the reported default threshold (10240) loads as unset."""
    result = convert_index_info_to_schema(svs_info(*tail))

    attrs = IndexSchema.from_dict(result).fields["emb"].attrs
    assert attrs.compression == compression
    assert attrs.training_threshold == threshold


def test_reported_default_threshold_matches_unset_target():
    """A live default (10240) must not make a target without it look different."""
    from redisvl.migration.utils import schemas_equal

    live = convert_index_info_to_schema(
        svs_info("compression", "GlobalSQ8", "training_threshold", 10240)
    )
    target = convert_index_info_to_schema(svs_info("compression", "GlobalSQ8"))
    assert schemas_equal(target, live, strip_excluded=True)


def test_convert_index_info_single_prefix():
    """Test converting index info with a single prefix.

    Single-element prefix lists are normalized to strings for backward compatibility.
    """
    index_info = {
        "index_name": "test_index",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["prefix_a"],
        ],
        "attributes": [],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_index"
    assert result["index"]["prefix"] == "prefix_a"  # normalized to string
    assert result["index"]["storage_type"] == "hash"


def test_convert_index_info_multiple_prefixes():
    """Test converting index info with multiple prefixes (issue #258)."""
    index_info = {
        "index_name": "test_index",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["prefix_a", "prefix_b", "prefix_c"],
        ],
        "attributes": [],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_index"
    assert result["index"]["prefix"] == ["prefix_a", "prefix_b", "prefix_c"]
    assert result["index"]["storage_type"] == "hash"


def test_convert_index_info_json_storage():
    """Test converting index info with JSON storage type.

    Single-element prefix lists are normalized to strings for backward compatibility.
    """
    index_info = {
        "index_name": "test_json_index",
        "index_definition": [
            "key_type",
            "JSON",
            "prefixes",
            ["json_prefix"],
        ],
        "attributes": [],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_json_index"
    assert result["index"]["prefix"] == "json_prefix"  # normalized to string
    assert result["index"]["storage_type"] == "json"


def test_convert_index_info_resp3_definition():
    """Test converting the RESP3 dictionary form returned by redis-py 8."""
    index_info = {
        "index_name": "test_resp3_index",
        "index_definition": {
            "key_type": "HASH",
            "prefixes": ["resp3_prefix"],
            "default_score": 1.0,
            "indexes_all": "false",
        },
        "attributes": [
            {
                "identifier": "category",
                "attribute": "category",
                "type": "TAG",
                "SEPARATOR": "|",
                "flags": ["CASESENSITIVE", "SORTABLE", "WITHSUFFIXTRIE"],
            }
        ],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_resp3_index"
    assert result["index"]["prefix"] == "resp3_prefix"
    assert result["index"]["storage_type"] == "hash"
    assert result["fields"] == [
        {
            "name": "category",
            "type": "tag",
            "attrs": {
                "case_sensitive": True,
                "sortable": True,
                "withsuffixtrie": True,
                "separator": "|",
            },
        }
    ]


def test_convert_index_info_resp2_withsuffixtrie():
    """Test the flag in the alternating-list FT.INFO form."""
    index_info = {
        "index_name": "test_resp2_index",
        "index_definition": ["key_type", "HASH", "prefixes", ["resp2_prefix"]],
        "attributes": [
            [
                "identifier",
                "title",
                "attribute",
                "title",
                "type",
                "TEXT",
                "WITHSUFFIXTRIE",
            ]
        ],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["fields"] == [
        {
            "name": "title",
            "type": "text",
            "attrs": {"withsuffixtrie": True},
        }
    ]


def test_convert_index_info_with_fields():
    """Test converting index info with field definitions."""
    index_info = {
        "index_name": "test_index",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["prefix_a", "prefix_b"],
        ],
        "attributes": [
            [
                "identifier",
                "user",
                "attribute",
                "user",
                "type",
                "TAG",
            ],
            [
                "identifier",
                "text",
                "attribute",
                "text",
                "type",
                "TEXT",
            ],
        ],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_index"
    assert result["index"]["prefix"] == ["prefix_a", "prefix_b"]
    assert len(result["fields"]) == 2
    assert result["fields"][0]["name"] == "user"
    assert result["fields"][0]["type"] == "tag"
    assert result["fields"][1]["name"] == "text"
    assert result["fields"][1]["type"] == "text"


def test_convert_index_info_stopwords_disabled():
    """Test converting index info with STOPWORDS 0 (disabled stopwords)."""
    index_info = {
        "index_name": "test_stopwords_disabled",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["test_sw:"],
        ],
        "attributes": [],
        "stopwords_list": [],  # STOPWORDS 0
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_stopwords_disabled"
    assert result["index"]["stopwords"] == []


def test_convert_index_info_custom_stopwords():
    """Test converting index info with custom stopwords list."""
    index_info = {
        "index_name": "test_custom_stopwords",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["test_csw:"],
        ],
        "attributes": [],
        "stopwords_list": [b"the", b"a", b"an"],  # Custom stopwords (as bytes)
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_custom_stopwords"
    assert result["index"]["stopwords"] == ["the", "a", "an"]


def test_convert_index_info_default_stopwords():
    """Test converting index info with default stopwords (no stopwords_list key).

    When no STOPWORDS clause is specified in FT.CREATE, Redis uses its default
    stopwords list, and FT.INFO does not include a stopwords_list key.
    """
    index_info = {
        "index_name": "test_default_stopwords",
        "index_definition": [
            "key_type",
            "HASH",
            "prefixes",
            ["test_dsw:"],
        ],
        "attributes": [],
        # No stopwords_list key - indicates default behavior
    }

    result = convert_index_info_to_schema(index_info)

    assert result["index"]["name"] == "test_default_stopwords"
    assert "stopwords" not in result["index"]  # Should not be present


def test_convert_index_info_resp2_unified_vector_flags():
    """Vector params packed into ``flags`` (redis-py 8, RESP2, legacy_responses=False)."""
    index_info = {
        "index_name": "test_index",
        "index_definition": {"key_type": "HASH", "prefixes": ["doc"]},
        "attributes": [
            {
                "identifier": "emb",
                "attribute": "emb",
                "type": "VECTOR",
                "flags": [
                    "algorithm",
                    "HNSW",
                    "data_type",
                    "FLOAT32",
                    "dim",
                    8,
                    "distance_metric",
                    "L2",
                    "M",
                    24,
                    "ef_construction",
                    300,
                    "INDEXMISSING",
                ],
            }
        ],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["fields"] == [
        {
            "name": "emb",
            "type": "vector",
            "attrs": {
                "dims": 8,
                "distance_metric": "l2",
                "algorithm": "hnsw",
                "datatype": "float32",
                "m": 24,
                "ef_construction": 300,
                "index_missing": True,
            },
        }
    ]


def test_convert_index_info_warns_on_unparseable_vector_field(caplog):
    """A vector field without params is skipped with a warning, not silently."""
    index_info = {
        "index_name": "test_index",
        "index_definition": {"key_type": "HASH", "prefixes": ["doc"]},
        "attributes": [
            {"identifier": "emb", "attribute": "emb", "type": "VECTOR", "flags": []}
        ],
    }

    with caplog.at_level("WARNING"):
        result = convert_index_info_to_schema(index_info)

    assert result["fields"] == []
    assert "'emb'" in caplog.text and "'test_index'" in caplog.text


def test_convert_index_info_resp2_unified_flags_skip_unknown_tokens():
    """An unknown bare token in ``flags`` does not shift the key/value pairs."""
    index_info = {
        "index_name": "test_index",
        "index_definition": {"key_type": "HASH", "prefixes": ["doc"]},
        "attributes": [
            {
                "identifier": "emb",
                "attribute": "emb",
                "type": "VECTOR",
                "flags": [
                    "algorithm",
                    "FLAT",
                    "data_type",
                    "FLOAT32",
                    "dim",
                    4,
                    "NEWMOD",
                    "distance_metric",
                    "IP",
                ],
            }
        ],
    }

    result = convert_index_info_to_schema(index_info)

    assert result["fields"][0]["attrs"]["distance_metric"] == "ip"
    assert result["fields"][0]["attrs"]["dims"] == 4


def test_convert_index_info_skips_vector_field_with_unparseable_dims(caplog):
    """A non-numeric ``dim`` skips the field with a warning instead of raising."""
    index_info = {
        "index_name": "test_index",
        "index_definition": {"key_type": "HASH", "prefixes": ["doc"]},
        "attributes": [
            {
                "identifier": "emb",
                "attribute": "emb",
                "type": "VECTOR",
                "flags": ["algorithm", "FLAT", "dim", "FUTUREMOD"],
            }
        ],
    }

    with caplog.at_level("WARNING"):
        result = convert_index_info_to_schema(index_info)

    assert result["fields"] == []
    assert "'emb'" in caplog.text


@pytest.mark.parametrize(
    "shape, rerank", [("pairs", None), ("mapping", "TRUE"), ("flags", "FALSE")]
)
def test_hnsw_info_round_trip(shape, rerank):
    field = {"identifier": "emb", "attribute": "emb", "type": "VECTOR"}
    params = {"algorithm": "HNSW", "dim": 4, "ef_runtime": 47}
    if rerank is not None:
        params["rerank"] = rerank
    pairs = [item for pair in params.items() for item in pair]
    if shape == "flags":
        field["flags"] = pairs
    elif shape == "mapping":
        field.update(params)
    else:
        field = [item for pair in field.items() for item in pair] + pairs

    info = {
        "index_name": "idx",
        "index_definition": ["key_type", "HASH", "prefixes", ["idx"]],
        "attributes": [field],
    }
    schema = IndexSchema.from_dict(convert_index_info_to_schema(info))
    args = schema.redis_fields[0].redis_args()
    assert args[args.index("EF_RUNTIME") + 1] == 47
    if rerank is None:
        assert "RERANK" not in args
    else:
        assert args[args.index("RERANK") + 1] == rerank
