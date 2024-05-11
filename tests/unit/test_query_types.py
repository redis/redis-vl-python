from redis.commands.search.query import Query
from redis.commands.search.result import Result

from redisvl.index.index import process_results
from redisvl.query import CountQuery, FilterQuery, RangeQuery, VectorQuery
from redisvl.query.filter import Tag

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

    # Test set_filter functionality
    new_filter_expression = Tag("category") == "Sportswear"
    count_query.set_filter(new_filter_expression)
    assert count_query.get_filter() == new_filter_expression

    fake_result = Result([2], "")
    assert process_results(fake_result, count_query, "json") == 2


def test_filter_query():
    # Create a filter expression
    filter_expression = Tag("brand") == "Nike"
    return_fields = ["brand", "price"]
    filter_query = FilterQuery(filter_expression, return_fields, 10)

    # Check properties
    assert filter_query._return_fields == return_fields
    assert filter_query._num_results == 10
    assert filter_query.get_filter() == filter_expression
    assert isinstance(filter_query.query, Query)
    assert isinstance(filter_query.params, dict)
    assert filter_query.params == {}
    assert filter_query._dialect == 2
    assert filter_query._sort_by == None

    # Test set_filter functionality
    new_filter_expression = Tag("category") == "Sportswear"
    filter_query.set_filter(new_filter_expression)
    assert filter_query.get_filter() == new_filter_expression

    # Test set_paging functionality
    filter_query.set_paging(5, 7)
    assert filter_query._first == 5
    assert filter_query._limit == 7
    assert filter_query._num_results == 10

    # Test sort_by functionality
    filter_query = FilterQuery(
        filter_expression, return_fields, num_results=10, sort_by="price"
    )
    assert filter_query._sort_by == "price"


def test_vector_query():
    # Create a vector query
    vector_query = VectorQuery(
        sample_vector, "vector_field", ["field1", "field2"], dialect=3, num_results=10
    )

    # Check properties
    assert vector_query._vector == sample_vector
    assert vector_query._field == "vector_field"
    assert vector_query._num_results == 10
    assert "field1" in vector_query._return_fields
    assert isinstance(vector_query.query, Query)
    assert isinstance(vector_query.params, dict)
    assert vector_query.params != {}
    assert vector_query._dialect == 3

    # Test set_filter functionality
    new_filter_expression = Tag("category") == "Sportswear"
    vector_query.set_filter(new_filter_expression)
    assert vector_query.get_filter() == new_filter_expression

    # Test set_paging functionality
    vector_query.set_paging(5, 7)
    assert vector_query._first == 5
    assert vector_query._limit == 7
    assert vector_query._num_results == 10


def test_range_query():
    # Create a filter expression
    filter_expression = Tag("brand") == "Nike"

    # Create a RangeQuery instance
    range_query = RangeQuery(
        sample_vector, "vector_field", ["field1"], filter_expression, num_results=10
    )

    # Check properties
    assert range_query._vector == sample_vector
    assert range_query._field == "vector_field"
    assert range_query._num_results == 10
    assert range_query.distance_threshold == 0.2
    assert "field1" in range_query._return_fields
    assert isinstance(range_query.query, Query)
    assert isinstance(range_query.params, dict)
    assert range_query.params != {}

    # Test set_filter functionality
    new_filter_expression = Tag("category") == "Outdoor"
    range_query.set_filter(new_filter_expression)
    assert range_query.get_filter() == new_filter_expression

    # Test set_paging functionality
    range_query.set_paging(5, 7)
    assert range_query._first == 5
    assert range_query._limit == 7
    assert range_query._num_results == 10
