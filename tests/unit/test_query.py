import pytest

from redis.commands.search.document import Document
from redis.commands.search.result import Result
from redis.commands.search.query import Query

from redisvl.query import CountQuery, FilterQuery, VectorQuery
from redisvl.index import process_results
from redisvl.query.filter import FilterExpression, Tag


# Sample data for testing
sample_vector = [0.1, 0.2, 0.3, 0.4]


# Test Cases

def test_count_query():
    # Create a filter expression
    filter_expression = Tag("brand") == "Nike"
    count_query = CountQuery(filter_expression)

    # Check properties
    assert isinstance(count_query.query, Query)
    assert isinstance(count_query.params, dict)
    assert count_query.params == {}

    fake_result = Result([2], "")
    assert process_results(fake_result, count_query, "json") == 2


def test_filter_query():
    # Create a filter expression
    filter_expression = Tag("brand") == "Nike"
    return_fields = ["brand", "price"]
    filter_query = FilterQuery(return_fields, filter_expression, 10)

    # Check properties
    assert filter_query._return_fields == return_fields
    assert filter_query._num_results == 10
    assert filter_query.get_filter() == filter_expression
    assert isinstance(filter_query.query, Query)
    assert isinstance(filter_query.params, dict)
    assert filter_query.params == {}


def test_vector_query():
    # Create a vector query
    vector_query = VectorQuery(sample_vector, "vector_field", ["field1", "field2"])

    # Check properties
    assert vector_query._vector == sample_vector
    assert vector_query._field == "vector_field"
    assert "field1" in vector_query._return_fields
    assert isinstance(vector_query.query, Query)
    assert isinstance(vector_query.params, dict)
    assert vector_query.params != {}
