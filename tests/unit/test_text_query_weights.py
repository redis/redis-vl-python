import pytest

from redisvl.query import TextQuery
from redisvl.query.filter import Tag


def test_text_query_accepts_weights_dict():
    """Test that TextQuery can accept a dictionary of field weights."""
    text = "example search query"

    # Dictionary with field names as keys and weights as values
    field_weights = {"title": 5.0, "content": 2.0, "tags": 1.0}

    # Should be able to create a TextQuery with weights dict
    text_query = TextQuery(text=text, text_field_name=field_weights, num_results=10)

    # The query should have a method to set field weights
    assert hasattr(text_query, "set_field_weights")

    # Check that weights are stored correctly
    assert text_query.field_weights == field_weights


def test_text_query_generates_weighted_query_string():
    """Test that TextQuery generates correct query string with field weights."""
    text = "search query"

    # Single field with weight > 1
    text_query = TextQuery(text=text, text_field_name={"title": 5.0}, num_results=10)

    query_string = str(text_query)
    # Should generate: @title:(search | query)=>{$weight:5.0}
    assert (
        "@title:(search | query)=>{ $weight: 5.0 }" in query_string
        or "@title:(search | query)=>{$weight:5.0}" in query_string
        or "@title:(search | query) => { $weight: 5.0 }" in query_string
    )


def test_text_query_multiple_fields_with_weights():
    """Test that TextQuery generates correct query string with multiple weighted fields."""
    text = "search terms"

    field_weights = {"title": 3.0, "content": 1.5, "tags": 1.0}

    text_query = TextQuery(text=text, text_field_name=field_weights, num_results=10)

    query_string = str(text_query)

    # Should generate query with all fields and their weights, combined with OR
    # The exact format depends on implementation, but all fields should be present
    assert "@title:" in query_string
    assert "@content:" in query_string
    assert "@tags:" in query_string

    # Weights should be in the query
    assert "$weight: 3.0" in query_string or "$weight:3.0" in query_string
    assert "$weight: 1.5" in query_string or "$weight:1.5" in query_string
    # Weight of 1.0 might be omitted as it's the default


def test_text_query_backward_compatibility():
    """Test that TextQuery still works with a single string field name."""
    text = "backward compatible"

    # Should work with just a string field name (original API)
    text_query = TextQuery(text=text, text_field_name="description", num_results=5)

    query_string = str(text_query)
    assert "@description:" in query_string
    assert "backward | compatible" in query_string

    # Field weights should have the single field with weight 1.0
    assert text_query.field_weights == {"description": 1.0}


def test_text_query_weight_validation():
    """Test that invalid weights are properly rejected."""
    text = "test query"

    # Test negative weight
    with pytest.raises(ValueError, match="must be positive"):
        TextQuery(text=text, text_field_name={"title": -1.0}, num_results=10)

    # Test zero weight
    with pytest.raises(ValueError, match="must be positive"):
        TextQuery(text=text, text_field_name={"title": 0}, num_results=10)

    # Test non-numeric weight
    with pytest.raises(TypeError, match="must be numeric"):
        TextQuery(text=text, text_field_name={"title": "five"}, num_results=10)

    # Test invalid field name type
    with pytest.raises(TypeError, match="Field name must be a string"):
        TextQuery(text=text, text_field_name={123: 1.0}, num_results=10)

    # Test invalid text_field_name type (not str or dict)
    with pytest.raises(
        TypeError, match="text_field_name must be a string or dictionary"
    ):
        TextQuery(text=text, text_field_name=["title", "content"], num_results=10)


def test_set_field_weights_method():
    """Test that set_field_weights method updates weights correctly."""
    text = "dynamic weights"

    # Start with single field
    text_query = TextQuery(text=text, text_field_name="title", num_results=10)

    assert text_query.field_weights == {"title": 1.0}

    # Update to multiple fields with weights
    new_weights = {"title": 5.0, "content": 2.0}
    text_query.set_field_weights(new_weights)

    assert text_query.field_weights == new_weights

    # Query string should reflect new weights
    query_string = str(text_query)
    assert "$weight: 5.0" in query_string or "$weight:5.0" in query_string
    assert "$weight: 2.0" in query_string or "$weight:2.0" in query_string
