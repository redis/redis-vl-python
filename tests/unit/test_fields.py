import pytest
from redis.commands.search.field import GeoField as RedisGeoField
from redis.commands.search.field import NumericField as RedisNumericField
from redis.commands.search.field import TagField as RedisTagField
from redis.commands.search.field import TextField as RedisTextField
from redis.commands.search.field import VectorField as RedisVectorField

from redisvl.schema.fields import (
    FieldFactory,
    FlatVectorField,
    GeoField,
    HNSWVectorField,
    NumericField,
    TagField,
    TextField,
)


# Utility functions to create schema instances with default values
def create_text_field_schema(**kwargs):
    defaults = {
        "name": "example_textfield",
        "attrs": {"sortable": False, "weight": 1.0},
    }
    defaults.update(kwargs)
    return TextField(**defaults)


def create_tag_field_schema(**kwargs):
    defaults = {
        "name": "example_tagfield",
        "attrs": {"sortable": False, "separator": ","},
    }
    defaults.update(kwargs)
    return TagField(**defaults)


def create_numeric_field_schema(**kwargs):
    defaults = {"name": "example_numericfield", "attrs": {"sortable": False}}
    defaults.update(kwargs)
    return NumericField(**defaults)


def create_geo_field_schema(**kwargs):
    defaults = {"name": "example_geofield", "attrs": {"sortable": False}}
    defaults.update(kwargs)
    return GeoField(**defaults)


def create_flat_vector_field(**kwargs):
    defaults = {
        "name": "example_flatvectorfield",
        "attrs": {"dims": 128, "algorithm": "FLAT"},
    }
    defaults["attrs"].update(kwargs)
    return FlatVectorField(**defaults)


def create_hnsw_vector_field(**kwargs):
    defaults = {
        "name": "example_hnswvectorfield",
        "attrs": {
            "dims": 128,
            "algorithm": "HNSW",
            "m": 16,
            "ef_construction": 200,
            "ef_runtime": 10,
            "epsilon": 0.01,
        },
    }
    defaults["attrs"].update(kwargs)
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
    field = schema.as_redis_field()
    assert isinstance(field, field_class)
    assert field.name == f"example_{field_class.__name__.lower()}"


def test_vector_fields_as_field():
    flat_vector_schema = create_flat_vector_field()
    flat_vector_field = flat_vector_schema.as_redis_field()
    assert isinstance(flat_vector_field, RedisVectorField)
    assert flat_vector_field.name == "example_flatvectorfield"

    hnsw_vector_schema = create_hnsw_vector_field()
    hnsw_vector_field = hnsw_vector_schema.as_redis_field()
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
    vector_field = vector_schema.as_redis_field()

    # Assert that the field is correctly created and the optional parameters are set.
    assert isinstance(vector_field, RedisVectorField)
    for param, value in extra_params.items():
        assert param.upper() in vector_field.args
        i = vector_field.args.index(param.upper())
        assert vector_field.args[i + 1] == value


def test_hnsw_vector_field_optional_params_not_set():
    # Create HNSW vector field without setting optional params
    hnsw_field = HNSWVectorField(
        name="example_vector", attrs={"dims": 128, "algorithm": "hnsw"}
    )

    assert hnsw_field.attrs.m == 16  # default value
    assert hnsw_field.attrs.ef_construction == 200  # default value
    assert hnsw_field.attrs.ef_runtime == 10  # default value
    assert hnsw_field.attrs.epsilon == 0.01  # default value

    field_exported = hnsw_field.as_redis_field()

    # Check the default values are correctly applied in the exported object
    assert field_exported.args[field_exported.args.index("M") + 1] == 16
    assert field_exported.args[field_exported.args.index("EF_CONSTRUCTION") + 1] == 200
    assert field_exported.args[field_exported.args.index("EF_RUNTIME") + 1] == 10
    assert field_exported.args[field_exported.args.index("EPSILON") + 1] == 0.01


def test_flat_vector_field_block_size_not_set():
    # Create Flat vector field without setting block_size
    flat_field = FlatVectorField(
        name="example_vector", attrs={"dims": 128, "algorithm": "flat"}
    )
    field_exported = flat_field.as_redis_field()

    # block_size and initial_cap should not be in the exported field if it was not set
    assert "BLOCK_SIZE" not in field_exported.args
    assert "INITIAL_CAP" not in field_exported.args


# Tests for standard field creation
@pytest.mark.parametrize(
    "field_type, expected_class",
    [
        ("tag", TagField),
        ("text", TextField),
        ("numeric", NumericField),
        ("geo", GeoField),
    ],
)
def test_create_standard_field(field_type, expected_class):
    field = FieldFactory.create_field(field_type, "example_field")
    assert isinstance(field, expected_class)
    assert field.name == "example_field"


# Tests for vector field creation
@pytest.mark.parametrize(
    "algorithm, expected_class",
    [
        ("flat", FlatVectorField),
        ("hnsw", HNSWVectorField),
    ],
)
def test_create_vector_field(algorithm, expected_class):
    field = FieldFactory.create_field(
        "vector", "example_vector_field", attrs={"algorithm": algorithm, "dims": 128}
    )
    assert isinstance(field, expected_class)
    assert field.name == "example_vector_field"


def test_create_vector_field_with_unknown_algorithm():
    """Test for unknown vector field algorithm."""
    with pytest.raises(ValueError) as e:
        FieldFactory.create_field(
            "vector",
            "example_vector_field",
            attrs={"algorithm": "unknown", "dims": 128},
        )
    assert "Unknown vector field algorithm" in str(e.value)


def test_missing_vector_field_algorithm():
    """Test for missing vector field algorithm."""
    with pytest.raises(ValueError) as e:
        FieldFactory.create_field("vector", "example_vector_field", attrs={"dims": 128})
    assert "Must provide algorithm param" in str(e.value)


def test_missing_vector_field_dims():
    """Test for missing vector field algorithm."""
    with pytest.raises(ValueError) as e:
        FieldFactory.create_field(
            "vector", "example_vector_field", attrs={"algorithm": "flat"}
        )
    assert "Must provide dims param" in str(e.value)


def test_create_unknown_field_type():
    """Test for unknown field type."""
    with pytest.raises(ValueError) as excinfo:
        FieldFactory.create_field("unknown", "example_field")
    assert "Unknown field type: unknown" in str(excinfo.value)


# Tests for new index_missing and index_empty attributes
def test_field_attributes_index_missing_and_empty():
    """Test the new index_missing and index_empty field attributes."""

    # Test TextField with both attributes
    text_field = TextField(
        name="description",
        attrs={"index_missing": True, "index_empty": True, "sortable": True},
    )
    assert text_field.attrs.index_missing == True
    assert text_field.attrs.index_empty == True
    assert text_field.attrs.sortable == True

    # Test TagField with both attributes
    tag_field = TagField(
        name="tags",
        attrs={"index_missing": True, "index_empty": True, "case_sensitive": True},
    )
    assert tag_field.attrs.index_missing == True
    assert tag_field.attrs.index_empty == True
    assert tag_field.attrs.case_sensitive == True

    # Test NumericField with index_missing only (index_empty not supported)
    num_field = NumericField(
        name="price", attrs={"index_missing": True, "sortable": True}
    )
    assert num_field.attrs.index_missing == True
    assert num_field.attrs.sortable == True

    # Test GeoField with index_missing only (index_empty not supported)
    geo_field = GeoField(name="location", attrs={"index_missing": True})
    assert geo_field.attrs.index_missing == True

    # Test vector fields with index_missing
    flat_vector_field = FlatVectorField(
        name="embedding",
        attrs={"algorithm": "flat", "dims": 128, "index_missing": True},
    )
    assert flat_vector_field.attrs.index_missing == True
    assert flat_vector_field.attrs.dims == 128

    hnsw_vector_field = HNSWVectorField(
        name="embedding2",
        attrs={"algorithm": "hnsw", "dims": 256, "index_missing": True},
    )
    assert hnsw_vector_field.attrs.index_missing == True
    assert hnsw_vector_field.attrs.dims == 256


def test_default_index_missing_and_empty_values():
    """Test that index_missing and index_empty default to False."""

    # Test default values for text field
    text_field = TextField(name="description")
    assert text_field.attrs.index_missing == False
    assert text_field.attrs.index_empty == False

    # Test default values for tag field
    tag_field = TagField(name="tags")
    assert tag_field.attrs.index_missing == False
    assert tag_field.attrs.index_empty == False

    # Test default values for numeric field
    num_field = NumericField(name="price")
    assert num_field.attrs.index_missing == False

    # Test default values for geo field
    geo_field = GeoField(name="location")
    assert geo_field.attrs.index_missing == False

    # Test default values for vector fields
    flat_vector_field = FlatVectorField(
        name="embedding", attrs={"algorithm": "flat", "dims": 128}
    )
    assert flat_vector_field.attrs.index_missing == False

    hnsw_vector_field = HNSWVectorField(
        name="embedding2", attrs={"algorithm": "hnsw", "dims": 256}
    )
    assert hnsw_vector_field.attrs.index_missing == False


@pytest.mark.parametrize(
    "field_class,field_name,extra_attrs,supports_index_empty",
    [
        (TextField, "text_field", {"weight": 2.0}, True),
        (TagField, "tag_field", {"separator": "|"}, True),
        (NumericField, "num_field", {"sortable": True}, False),
        (GeoField, "geo_field", {"sortable": True}, False),
    ],
)
def test_redis_field_creation_with_index_attributes(
    field_class, field_name, extra_attrs, supports_index_empty
):
    """Test that index_missing and index_empty are properly passed to Redis field objects."""

    # Test with index_missing=True
    attrs = {"index_missing": True}
    attrs.update(extra_attrs)

    if supports_index_empty:
        attrs["index_empty"] = True

    field = field_class(name=field_name, attrs=attrs)
    redis_field = field.as_redis_field()

    # Check that the field was created successfully
    assert redis_field.name == field_name

    # For Redis fields, these attributes would be passed as keyword arguments
    # We can't directly inspect them, but we can verify the field creation doesn't fail


def test_vector_fields_redis_creation_with_index_missing():
    """Test that vector fields properly handle index_missing in Redis field creation."""

    # Test FlatVectorField with index_missing
    flat_field = FlatVectorField(
        name="flat_embedding",
        attrs={
            "algorithm": "flat",
            "dims": 128,
            "index_missing": True,
            "block_size": 100,
        },
    )
    redis_field = flat_field.as_redis_field()
    assert isinstance(redis_field, RedisVectorField)
    assert redis_field.name == "flat_embedding"

    # Test HNSWVectorField with index_missing
    hnsw_field = HNSWVectorField(
        name="hnsw_embedding",
        attrs={"algorithm": "hnsw", "dims": 256, "index_missing": True, "m": 24},
    )
    redis_field = hnsw_field.as_redis_field()
    assert isinstance(redis_field, RedisVectorField)
    assert redis_field.name == "hnsw_embedding"


def test_vector_field_data_includes_index_missing():
    """Test that vector field field_data includes INDEXMISSING when enabled."""

    # Test with index_missing=True
    flat_field_with_missing = FlatVectorField(
        name="embedding",
        attrs={"algorithm": "flat", "dims": 128, "index_missing": True},
    )
    field_data = flat_field_with_missing.attrs.field_data
    assert "INDEXMISSING" in field_data
    assert field_data["INDEXMISSING"] == True

    # Test with index_missing=False (default)
    flat_field_without_missing = FlatVectorField(
        name="embedding", attrs={"algorithm": "flat", "dims": 128}
    )
    field_data = flat_field_without_missing.attrs.field_data
    assert "INDEXMISSING" not in field_data

    # Test HNSW field with index_missing=True
    hnsw_field_with_missing = HNSWVectorField(
        name="embedding",
        attrs={"algorithm": "hnsw", "dims": 256, "index_missing": True},
    )
    field_data = hnsw_field_with_missing.attrs.field_data
    assert "INDEXMISSING" in field_data
    assert field_data["INDEXMISSING"] == True


def test_field_factory_with_new_attributes():
    """Test FieldFactory.create_field with the new index attributes."""

    # Test creating TextField with new attributes
    text_field = FieldFactory.create_field(
        "text", "description", attrs={"index_missing": True, "index_empty": True}
    )
    assert isinstance(text_field, TextField)
    assert text_field.attrs.index_missing == True
    assert text_field.attrs.index_empty == True

    # Test creating TagField with new attributes
    tag_field = FieldFactory.create_field(
        "tag", "categories", attrs={"index_missing": True, "index_empty": True}
    )
    assert isinstance(tag_field, TagField)
    assert tag_field.attrs.index_missing == True
    assert tag_field.attrs.index_empty == True

    # Test creating NumericField with index_missing
    num_field = FieldFactory.create_field(
        "numeric", "price", attrs={"index_missing": True}
    )
    assert isinstance(num_field, NumericField)
    assert num_field.attrs.index_missing == True

    # Test creating vector field with index_missing
    vector_field = FieldFactory.create_field(
        "vector",
        "embedding",
        attrs={"algorithm": "flat", "dims": 128, "index_missing": True},
    )
    assert isinstance(vector_field, FlatVectorField)
    assert vector_field.attrs.index_missing == True
