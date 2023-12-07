import pathlib

import pytest
from pydantic import ValidationError

from redis.commands.search.field import (
    GeoField as RedisGeoField,
    NumericField as RedisNumericField,
    TagField as RedisTagField,
    TextField as RedisTextField,
    VectorField as RedisVectorField,
)

from redisvl.schema.fields import (
    FlatVectorField,
    GeoField,
    HNSWVectorField,
    TagField,
    TextField,
    NumericField,
)

from redisvl.schema.schema import (
    IndexModel,
    Schema,
    StorageType,
    read_schema,
)


def get_base_path():
    return pathlib.Path(__file__).parent.resolve()


# Utility functions to create schema instances with default values
def create_text_field_schema(**kwargs):
    defaults = {"name": "example_textfield", "sortable": False, "weight": 1.0}
    defaults.update(kwargs)
    return TextField(**defaults)


def create_tag_field_schema(**kwargs):
    defaults = {"name": "example_tagfield", "sortable": False, "separator": ","}
    defaults.update(kwargs)
    return TagField(**defaults)


def create_numeric_field_schema(**kwargs):
    defaults = {"name": "example_numericfield", "sortable": False}
    defaults.update(kwargs)
    return NumericField(**defaults)


def create_geo_field_schema(**kwargs):
    defaults = {"name": "example_geofield", "sortable": False}
    defaults.update(kwargs)
    return GeoField(**defaults)


def create_flat_vector_field(**kwargs):
    defaults = {"name": "example_flatvectorfield", "dims": 128, "algorithm": "FLAT"}
    defaults.update(kwargs)
    return FlatVectorField(**defaults)


def create_hnsw_vector_field(**kwargs):
    defaults = {
        "name": "example_hnswvectorfield",
        "dims": 128,
        "algorithm": "HNSW",
        "m": 16,
        "ef_construction": 200,
        "ef_runtime": 10,
        "epsilon": 0.01,
    }
    defaults.update(kwargs)
    return HNSWVectorField(**defaults)


# Tests for field schema creation and validation
@pytest.mark.parametrize(
    "schema_func,field_class",
    [
        (create_text_field_schema, TextField),
        (create_tag_field_schema, TagField),
        (create_numeric_field_schema, NumericField),
        (create_geo_field_schema, GeoField),
    ],
)
def test_field_schema_as_field(schema_func, field_class):
    schema = schema_func()
    field = schema.as_field()
    assert isinstance(field, field_class)
    assert field.name == f"example_{field_class.__name__.lower()}"


def test_vector_fields_as_field():
    flat_vector_schema = create_flat_vector_field()
    flat_vector_field = flat_vector_schema.as_field()
    assert isinstance(flat_vector_field, RedisVectorField)
    assert flat_vector_field.name == "example_flatvectorfield"

    hnsw_vector_schema = create_hnsw_vector_field()
    hnsw_vector_field = hnsw_vector_schema.as_field()
    assert isinstance(hnsw_vector_field, RedisVectorField)
    assert hnsw_vector_field.name == "example_hnswvectorfield"


@pytest.mark.parametrize(
    "vector_schema_func,extra_params",
    [
        (create_flat_vector_field, {"block_size": 100}),
        (create_hnsw_vector_field, {"m": 24, "ef_construction": 300}),
    ],
)
def test_vector_fields_with_optional_params(vector_schema_func, extra_params):
    # Create a vector schema with additional parameters set.
    vector_schema = vector_schema_func(**extra_params)
    vector_field = vector_schema.as_field()

    # Assert that the field is correctly created and the optional parameters are set.
    assert isinstance(vector_field, RedisVectorField)
    for param, value in extra_params.items():
        assert param.upper() in vector_field.args
        i = vector_field.args.index(param.upper())
        assert vector_field.args[i + 1] == value


def test_hnsw_vector_field_optional_params_not_set():
    # Create HNSW vector field without setting optional params
    hnsw_field = HNSWVectorField(name="example_vector", dims=128, algorithm="HNSW")

    assert hnsw_field.m == 16  # default value
    assert hnsw_field.ef_construction == 200  # default value
    assert hnsw_field.ef_runtime == 10  # default value
    assert hnsw_field.epsilon == 0.01  # default value

    field_exported = hnsw_field.as_field()

    # Check the default values are correctly applied in the exported object
    assert field_exported.args[field_exported.args.index("M") + 1] == 16
    assert field_exported.args[field_exported.args.index("EF_CONSTRUCTION") + 1] == 200
    assert field_exported.args[field_exported.args.index("EF_RUNTIME") + 1] == 10
    assert field_exported.args[field_exported.args.index("EPSILON") + 1] == 0.01


def test_flat_vector_field_block_size_not_set():
    # Create Flat vector field without setting block_size
    flat_field = FlatVectorField(name="example_vector", dims=128, algorithm="FLAT")
    field_exported = flat_field.as_field()

    # block_size and initial_cap should not be in the exported field if it was not set
    assert "BLOCK_SIZE" not in field_exported.args
    assert "INITIAL_CAP" not in field_exported.args


# Tests for IndexModel


def test_index_model_defaults():
    index = IndexModel(name="test_index")
    assert index.name == "test_index"
    assert index.prefix == "rvl"
    assert index.key_separator == ":"
    assert index.storage_type == StorageType.HASH


def test_index_model_custom_settings():
    index = IndexModel(
        name="test_index", prefix="custom", key_separator="_", storage_type="json"
    )
    assert index.name == "test_index"
    assert index.prefix == "custom"
    assert index.key_separator == "_"
    assert index.storage_type == StorageType.JSON


def test_index_model_validation_errors():
    # Missing required field
    with pytest.raises(ValueError):
        IndexModel()

    # Invalid type
    with pytest.raises(ValidationError):
        IndexModel(name="test_index", prefix=None)

    # Invalid type
    with pytest.raises(ValidationError):
        IndexModel(name="test_index", key_separator=None)

    # Invalid type
    with pytest.raises(ValidationError):
        IndexModel(name="test_index", storage_type=None)


def test_schema_model_validation_failures():
    # Invalid storage type
    with pytest.raises(ValueError):
        invalid_index = {"name": "test_index", "storage_type": "unsupported"}
        Schema(index=invalid_index, fields={})

    # Missing required field
    with pytest.raises(ValueError):
        Schema(index={}, fields={})

    # Invalid index
    with pytest.raises(ValueError):
        Schema(index=12, fields={})


def test_read_hash_schema():
    hash_schema = read_schema(
        str(get_base_path().joinpath("../sample_hash_schema.yaml"))
    )
    assert hash_schema.index_name == hash_schema._index.name == "hash-test"


def test_read_json_schema():
    json_schema = read_schema(
        str(get_base_path().joinpath("../sample_json_schema.yaml"))
    )
    assert json_schema.index_name == json_schema._index.name == "json-test"


def test_read_schema_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_schema("non_existent_file.yaml")

