import pytest
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
    assert isinstance(count_query, Query)
    assert isinstance(count_query.query, Query)
    assert isinstance(count_query.params, dict)
    assert count_query.params == {}

    # Test set_filter functionality
    new_filter_expression = Tag("category") == "Sportswear"
    count_query.set_filter(new_filter_expression)
    assert count_query.filter == new_filter_expression

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
    assert filter_query.filter == filter_expression
    assert isinstance(filter_query, Query)
    assert isinstance(filter_query.query, Query)
    assert isinstance(filter_query.params, dict)
    assert filter_query.params == {}
    assert filter_query._dialect == 2
    assert filter_query._sortby == None
    assert filter_query._in_order == False

    # Test set_filter functionality
    new_filter_expression = Tag("category") == "Sportswear"
    filter_query.set_filter(new_filter_expression)
    assert filter_query.filter == new_filter_expression

    # Test paging functionality
    filter_query.paging(5, 7)
    assert filter_query._offset == 5
    assert filter_query._num == 7
    assert filter_query._num_results == 10

    # Test sort_by functionality
    filter_query = FilterQuery(
        filter_expression, return_fields, num_results=10, sort_by="price"
    )
    assert filter_query._sortby is not None

    # Test in_order functionality
    filter_query = FilterQuery(
        filter_expression, return_fields, num_results=10, in_order=True
    )
    assert filter_query._in_order


def test_vector_query():
    # Create a vector query
    vector_query = VectorQuery(
        sample_vector, "vector_field", ["field1", "field2"], dialect=3, num_results=10
    )

    # Check properties
    assert vector_query._vector == sample_vector
    assert vector_query._vector_field_name == "vector_field"
    assert vector_query._num_results == 10
    assert vector_query._return_fields == ["field1", "field2", "vector_distance"]
    assert isinstance(vector_query, Query)
    assert isinstance(vector_query.query, Query)
    assert isinstance(vector_query.params, dict)
    assert vector_query.params != {}
    assert vector_query._dialect == 3
    assert vector_query._sortby.args[0] == VectorQuery.DISTANCE_ID
    assert vector_query._in_order == False

    # Test set_filter functionality
    new_filter_expression = Tag("category") == "Sportswear"
    vector_query.set_filter(new_filter_expression)
    assert vector_query.filter == new_filter_expression

    # Test paging functionality
    vector_query.paging(5, 7)
    assert vector_query._offset == 5
    assert vector_query._num == 7
    assert vector_query._num_results == 10

    # Test sort_by functionality
    vector_query = VectorQuery(
        sample_vector,
        "vector_field",
        ["field1", "field2"],
        dialect=3,
        num_results=10,
        sort_by="field2",
    )
    assert vector_query._sortby.args[0] == "field2"

    # Test in_order functionality
    vector_query = VectorQuery(
        sample_vector,
        "vector_field",
        ["field1", "field2"],
        dialect=3,
        num_results=10,
        in_order=True,
    )
    assert vector_query._in_order


def test_range_query():
    # Create a filter expression
    filter_expression = Tag("brand") == "Nike"

    # Create a RangeQuery instance
    range_query = RangeQuery(
        sample_vector, "vector_field", ["field1"], filter_expression, num_results=10
    )

    # Check properties
    assert range_query._vector == sample_vector
    assert range_query._vector_field_name == "vector_field"
    assert range_query._num_results == 10
    assert range_query.distance_threshold == 0.2
    assert "field1" in range_query._return_fields
    assert isinstance(range_query, Query)
    assert isinstance(range_query.query, Query)
    assert isinstance(range_query.params, dict)
    assert range_query.params != {}
    assert range_query._sortby.args[0] == RangeQuery.DISTANCE_ID

    # Test set_distance_threshold functionality
    range_query.set_distance_threshold(0.1)
    assert range_query.distance_threshold == 0.1

    # Test set_filter functionality
    new_filter_expression = Tag("category") == "Outdoor"
    range_query.set_filter(new_filter_expression)
    assert range_query.filter == new_filter_expression

    # Test paging functionality
    range_query.paging(5, 7)
    assert range_query._offset == 5
    assert range_query._num == 7
    assert range_query._num_results == 10

    # Test sort_by functionality
    range_query = RangeQuery(
        sample_vector,
        "vector_field",
        ["field1"],
        filter_expression,
        num_results=10,
        sort_by="field1",
    )
    assert range_query._sortby.args[0] == "field1"

    # Test in_order functionality
    range_query = RangeQuery(
        sample_vector,
        "vector_field",
        ["field1"],
        filter_expression,
        num_results=10,
        in_order=True,
    )
    assert range_query._in_order


@pytest.mark.parametrize(
    "query",
    [
        CountQuery(),
        FilterQuery(),
        VectorQuery(vector=[1, 2, 3], vector_field_name="vector"),
        RangeQuery(vector=[1, 2, 3], vector_field_name="vector"),
    ],
)
def test_query_modifiers(query):
    query.paging(3, 5)
    assert query._offset == 3
    assert query._num == 5

    query.dialect(4)
    assert query._dialect == 4

    query.in_order()
    assert query._in_order

    query.sort_by("time")
    assert query._sortby.args[0] == "time"

    query.scorer("BM25")
    assert query._scorer == "BM25"

    query.timeout(20)
    assert query._timeout == 20

    query.slop(10)
    assert query._slop == 10

    query.verbatim()
    assert query._verbatim

    query.no_content()
    assert query._no_content

    query.no_stopwords()
    assert query._no_stopwords

    query.with_scores()
    assert query._with_scores

    query.limit_fields("test")
    assert query._fields == ("test",)

    f = Tag("test") == "foo"
    query.set_filter(f)
    assert query._filter_expression == f

    # double check all other states
    assert query._offset == 3
    assert query._num == 5
    assert query._dialect == 4
    assert query._in_order
    assert query._sortby.args[0] == "time"
    assert query._scorer == "BM25"
    assert query._timeout == 20
    assert query._slop == 10
    assert query._verbatim
    assert query._no_content
    assert query._no_stopwords
    assert query._with_scores
    assert query._fields == ("test",)
