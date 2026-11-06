"""Pin the visible string form of RedisVL's string enums.

These enums are ``StrEnum`` subclasses, so ``str()`` and f-strings give the
member's value. Serialisation (pydantic, YAML, the FT.CREATE arguments sent to
Redis) went through the value before the move from ``(str, Enum)`` and must
not change.
"""

from enum import StrEnum

import pytest
import yaml
from redis.connection import Encoder

from redisvl.extensions.message_history.schema import ChatRole
from redisvl.mcp.errors import MCPErrorCode
from redisvl.query.filter import Inclusive, Num, Tag
from redisvl.query.query import HybridPolicy
from redisvl.schema import IndexSchema
from redisvl.schema.fields import (
    CompressionType,
    FieldTypes,
    VectorDataType,
    VectorDistanceMetric,
    VectorIndexAlgorithm,
)

STR_ENUMS = [
    MCPErrorCode,
    ChatRole,
    FieldTypes,
    VectorDistanceMetric,
    VectorDataType,
    VectorIndexAlgorithm,
    CompressionType,
    HybridPolicy,
    Inclusive,
]


@pytest.mark.parametrize(
    "member", [m for cls in STR_ENUMS for m in cls], ids=lambda m: repr(m)
)
def test_member_stringifies_to_its_value(member):
    assert isinstance(member, StrEnum)
    assert str(member) == f"{member}" == "%s" % member == member.value


def _schema(**index):
    return IndexSchema.from_dict(
        {
            "index": {"name": "enums", "prefix": "enums", **index},
            "fields": [
                {"name": "tag", "type": "tag"},
                {
                    "name": "flat",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 4,
                        "datatype": "int8",
                        "distance_metric": "l2",
                    },
                },
                {
                    "name": "svs",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "svs-vamana",
                        "dims": 8,
                        "datatype": "float16",
                        "distance_metric": "ip",
                        "compression": "LeanVec4x8",
                        "reduce": 4,
                    },
                },
            ],
        }
    )


def test_ft_create_arguments_encode_member_values():
    encoder = Encoder("utf-8", "strict", False)
    fields = _schema(storage_type="json").fields

    def encoded(name):
        return [encoder.encode(a) for a in fields[name].as_redis_field().redis_args()]

    assert encoded("tag")[3] == b"TAG"
    assert encoded("flat") == [
        *(b"$.flat", b"AS", b"flat", b"VECTOR", b"FLAT", b"6"),
        *(b"TYPE", b"INT8", b"DIM", b"4", b"DISTANCE_METRIC", b"L2"),
    ]
    svs = encoded("svs")
    assert svs[3:5] == [b"VECTOR", b"SVS-VAMANA"]
    assert svs[svs.index(b"TYPE") + 1] == b"FLOAT16"
    assert svs[svs.index(b"DISTANCE_METRIC") + 1] == b"IP"
    assert svs[svs.index(b"COMPRESSION") + 1] == b"LeanVec4x8"


def test_schema_serialisation_uses_member_values(tmp_path):
    schema = _schema()

    flat = schema.model_dump(mode="json")["fields"]["flat"]
    assert flat["type"] == "vector"
    assert flat["attrs"]["algorithm"] == "FLAT"
    assert flat["attrs"]["datatype"] == "INT8"
    assert flat["attrs"]["distance_metric"] == "L2"
    svs_attrs = schema.fields["svs"].attrs
    assert svs_attrs.model_dump(mode="json")["compression"] == "LeanVec4x8"

    as_dict = schema.to_dict()
    schema.to_yaml(str(tmp_path / "schema.yaml"))
    assert yaml.safe_load((tmp_path / "schema.yaml").read_text()) == as_dict
    flat_attrs = next(f for f in as_dict["fields"] if f["name"] == "flat")["attrs"]
    assert flat_attrs["algorithm"] == "flat"
    assert flat_attrs["datatype"] == "int8"
    assert flat_attrs["distance_metric"] == "l2"


def test_error_messages_name_the_value():
    with pytest.raises(ValueError, match=r"Got: INT8\. "):
        IndexSchema.from_dict(
            {
                "index": {"name": "e", "prefix": "e"},
                "fields": [
                    {
                        "name": "v",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "svs-vamana",
                            "dims": 4,
                            "datatype": "int8",
                        },
                    }
                ],
            }
        )


def test_query_strings_use_member_values():
    assert str(Tag("role") == [ChatRole.USER, ChatRole.TOOL]) == "@role:{user|tool}"
    assert str(Num("n").between(1, 2, inclusive=Inclusive.LEFT)) == "@n:[1 (2]"
