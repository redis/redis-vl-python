import typing as t
from pathlib import Path

import yaml
from redis.commands.search.field import (
    GeoField,
    NumericField,
    TagField,
    TextField,
    VectorField,
)

from redisvl.utils.log import get_logger

logger = get_logger(__name__)


def read_schema(file_path: str):
    fp = Path(file_path).resolve()
    if not fp.exists():
        logger.error(f"Schema file {file_path} does not exist")
        raise FileNotFoundError(f"Schema file {file_path} does not exist")

    with open(fp, "r") as f:
        schema = yaml.safe_load(f)

    try:
        index_schema = schema["index"]
        fields_schema = schema["fields"]
    except KeyError:
        logger.error("Schema file must contain both a 'fields' and 'index' key")
        raise

    index_attrs = read_index_spec(index_schema)
    fields = read_field_spec(fields_schema)
    return index_attrs, fields


def read_index_spec(index_spec: t.Dict[str, t.Any]):
    """Read index specification and return the fields

    Args:
        index_schema (dict): Index specification from schema file.

    Returns:
        index_fields (dict): List of index fields.
    """
    # TODO parsing and validation here
    return index_spec


def read_field_spec(field_spec: t.Dict[str, t.Any]):
    """
    Read a schema file and return a list of RediSearch fields.

    Args:
        field_schema (dict): Field specification from schema file.

    Returns:
        fields: list of RediSearch fields.
    """
    fields = []
    for key, field in field_spec.items():
        if key.upper() == "TAG":
            for name, attrs in field.items():
                fields.append(TagField(name, **attrs))
        elif key.upper() == "VECTOR":
            for name, attrs in field.items():
                fields.append(_create_vector_field(name, **attrs))
        elif key.upper() == "GEO":
            for name, attrs in field.items():
                fields.append(GeoField(name, **attrs))
        elif key.upper() == "TEXT":
            for name, attrs in field.items():
                fields.append(TextField(name, **attrs))
        elif key.upper() == "NUMERIC":
            for name, attrs in field.items():
                fields.append(NumericField(name, **attrs))
        else:
            logger.error(f"Invalid field type: {key}")
            raise ValueError(f"Invalid field type: {key}")
    return fields


def _create_vector_field(
    name: str,
    dims: int,
    algorithm: str = "FLAT",
    datatype: str = "FLOAT32",
    distance_metric: str = "COSINE",
    initial_cap: int = 1000000,
    block_size: int = 1000,
    m: int = 16,
    ef_construction: int = 200,
    ef_runtime: int = 10,
    epsilon: float = 0.8,
):
    """Create a RediSearch VectorField.

    Args:
      name: The name of the field.
      algorithm: The algorithm used to index the vector.
      dims: The dimensionality of the vector.
      datatype: The type of the vector. default: FLOAT32
      distance_metric: The distance metric used to compare vectors.
      initial_cap: The initial capacity of the index.
      block_size: The block size of the index.
      m: The number of outgoing edges in the HNSW graph.
      ef_construction: Number of maximum allowed potential outgoing edges
                       candidates for each node in the graph, during the graph building.
      ef_runtime: The umber of maximum top candidates to hold during the KNN search

    returns:
      A RediSearch VectorField.
    """
    if algorithm.upper() == "HNSW":
        return VectorField(
            name,
            "HNSW",
            {
                "TYPE": datatype.upper(),
                "DIM": dims,
                "DISTANCE_METRIC": distance_metric.upper(),
                "INITIAL_CAP": initial_cap,
                "M": m,
                "EF_CONSTRUCTION": ef_construction,
                "EF_RUNTIME": ef_runtime,
                "EPSILON": epsilon,
            },
        )
    else:
        return VectorField(
            name,
            "FLAT",
            {
                "TYPE": datatype.upper(),
                "DIM": dims,
                "DISTANCE_METRIC": distance_metric.upper(),
                "INITIAL_CAP": initial_cap,
                "BLOCK_SIZE": block_size,
            },
        )
