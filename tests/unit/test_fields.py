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
    BaseField,
    FlatVectorField,
    GeoField,
    HNSWVectorField,
    TagField,
    TextField,
    NumericField,
    FieldFactory
)



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
        (create_text_field_schema, RedisTextField),
        (create_tag_field_schema, RedisTagField),
        (create_numeric_field_schema, RedisNumericField),
        (create_geo_field_schema, RedisGeoField),
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


# Tests for standard field creation
@pytest.mark.parametrize("field_type, expected_class", [
    ("tag", TagField),
    ("text", TextField),
    ("numeric", NumericField),
    ("geo", GeoField),
])
def test_create_standard_field(field_type, expected_class):
    field = FieldFactory.create_field(field_type, "example_field")
    assert isinstance(field, expected_class)
    assert field.name == "example_field"

# Tests for vector field creation
@pytest.mark.parametrize("algorithm, expected_class", [
    ("flat", FlatVectorField),
    ("hnsw", HNSWVectorField),
])
def test_create_vector_field(algorithm, expected_class):
    field = FieldFactory.create_field("vector", "example_vector_field", algorithm=algorithm, dims=128)
    assert isinstance(field, expected_class)
    assert field.name == "example_vector_field"

def test_create_vector_field_with_unknown_algorithm():
    """Test for unknown vector field algorithm."""
    with pytest.raises(ValueError) as e:
        FieldFactory.create_field("vector", "example_vector_field", algorithm="unknown", dims=128)
    assert "Unknown vector field algorithm" in str(e.value)

def test_missing_vector_field_algorithm():
    """Test for missing vector field algorithm."""
    with pytest.raises(ValueError) as e:
        FieldFactory.create_field("vector", "example_vector_field", dims=128)
    assert "Must provide algorithm param" in str(e.value)

def test_missing_vector_field_dims():
    """Test for missing vector field algorithm."""
    with pytest.raises(ValueError) as e:
        FieldFactory.create_field("vector", "example_vector_field", algorithm="flat")
    assert "Must provide dims param" in str(e.value)

def test_create_unknown_field_type():
    """Test for unknown field type."""
    with pytest.raises(ValueError) as excinfo:
        FieldFactory.create_field("unknown", "example_field")
    assert "Unknown field type: unknown" in str(excinfo.value)

