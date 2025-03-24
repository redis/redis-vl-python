import pytest
from redis.commands.search.query import Query
from redis.commands.search.result import Result

from redisvl.index.index import process_results
from redisvl.query import CountQuery, FilterQuery, RangeQuery, VectorQuery
from redisvl.query.filter import Tag
from redisvl.query.query import VectorRangeQuery

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
    assert filter_query._sortby is None
    assert filter_query._in_order is False

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
    assert vector_query._in_order is False

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


@pytest.mark.parametrize(
    "query",
    [
        CountQuery(),
        FilterQuery(),
        VectorQuery(vector=[1, 2, 3], vector_field_name="vector"),
        RangeQuery(vector=[1, 2, 3], vector_field_name="vector"),
    ],
)
def test_string_filter_expressions(query):
    # No filter
    query.set_filter("*")
    assert query._filter_expression == "*"

    # Simple full text search
    query.set_filter("hello world")
    assert query._filter_expression == "hello world"
    assert query.query_string().__contains__("hello world")

    # Optional flag
    query.set_filter("~(@desciption:(hello | world))")
    assert query._filter_expression == "~(@desciption:(hello | world))"
    assert query.query_string().__contains__("~(@desciption:(hello | world))")


def test_vector_query_hybrid_policy():
    """Test that VectorQuery correctly handles hybrid policy parameters."""
    # Create a vector query with hybrid policy
    vector_query = VectorQuery(
        [0.1, 0.2, 0.3, 0.4], "vector_field", hybrid_policy="BATCHES"
    )

    # Check properties
    assert vector_query.hybrid_policy == "BATCHES"
    assert vector_query.batch_size is None

    # Check query string
    query_string = str(vector_query)
    assert "HYBRID_POLICY BATCHES" in query_string

    # Test with batch size
    vector_query = VectorQuery(
        [0.1, 0.2, 0.3, 0.4], "vector_field", hybrid_policy="BATCHES", batch_size=50
    )

    # Check properties
    assert vector_query.hybrid_policy == "BATCHES"
    assert vector_query.batch_size == 50

    # Check query string
    query_string = str(vector_query)
    assert "HYBRID_POLICY BATCHES BATCH_SIZE 50" in query_string

    # Test with ADHOC_BF policy
    vector_query = VectorQuery(
        [0.1, 0.2, 0.3, 0.4], "vector_field", hybrid_policy="ADHOC_BF"
    )

    # Check properties
    assert vector_query.hybrid_policy == "ADHOC_BF"

    # Check query string
    query_string = str(vector_query)
    assert "HYBRID_POLICY ADHOC_BF" in query_string


def test_vector_query_set_hybrid_policy():
    """Test that VectorQuery setter methods work properly."""
    # Create a vector query
    vector_query = VectorQuery([0.1, 0.2, 0.3, 0.4], "vector_field")

    # Initially no hybrid policy
    assert vector_query.hybrid_policy is None
    assert "HYBRID_POLICY" not in str(vector_query)

    # Set hybrid policy
    vector_query.set_hybrid_policy("BATCHES")

    # Check properties
    assert vector_query.hybrid_policy == "BATCHES"

    # Check query string
    query_string = str(vector_query)
    assert "HYBRID_POLICY BATCHES" in query_string

    # Set batch size
    vector_query.set_batch_size(100)

    # Check properties
    assert vector_query.batch_size == 100

    # Check query string
    query_string = str(vector_query)
    assert "HYBRID_POLICY BATCHES BATCH_SIZE 100" in query_string


def test_vector_query_invalid_hybrid_policy():
    """Test error handling for invalid hybrid policy values."""
    # Test with invalid hybrid policy
    with pytest.raises(ValueError, match=r"hybrid_policy must be one of.*"):
        VectorQuery([0.1, 0.2, 0.3, 0.4], "vector_field", hybrid_policy="INVALID")

    # Create a valid vector query
    vector_query = VectorQuery([0.1, 0.2, 0.3, 0.4], "vector_field")

    # Test with invalid hybrid policy
    with pytest.raises(ValueError, match=r"hybrid_policy must be one of.*"):
        vector_query.set_hybrid_policy("INVALID")

    # Test with invalid batch size types
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        vector_query.set_batch_size("50")

    # Test with invalid batch size values
    with pytest.raises(ValueError, match="batch_size must be positive"):
        vector_query.set_batch_size(0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        vector_query.set_batch_size(-10)


def test_vector_range_query_epsilon():
    """Test that VectorRangeQuery correctly handles epsilon parameter."""
    # Create a range query with epsilon
    range_query = VectorRangeQuery(
        [0.1, 0.2, 0.3, 0.4], "vector_field", epsilon=0.05, distance_threshold=0.3
    )

    # Check properties
    assert range_query.epsilon == 0.05
    assert range_query.distance_threshold == 0.3

    # Check query string
    query_string = str(range_query)
    assert "$EPSILON: 0.05" in query_string

    # Test setting epsilon
    range_query.set_epsilon(0.1)
    assert range_query.epsilon == 0.1
    assert "$EPSILON: 0.1" in str(range_query)


def test_vector_range_query_invalid_epsilon():
    """Test error handling for invalid epsilon values."""
    # Test with invalid epsilon type
    with pytest.raises(TypeError, match="epsilon must be of type float or int"):
        VectorRangeQuery([0.1, 0.2, 0.3, 0.4], "vector_field", epsilon="0.05")

    # Test with negative epsilon
    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        VectorRangeQuery([0.1, 0.2, 0.3, 0.4], "vector_field", epsilon=-0.05)

    # Create a valid range query
    range_query = VectorRangeQuery([0.1, 0.2, 0.3, 0.4], "vector_field")

    # Test with invalid epsilon
    with pytest.raises(TypeError, match="epsilon must be of type float or int"):
        range_query.set_epsilon("0.05")

    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        range_query.set_epsilon(-0.05)


def test_vector_range_query_construction():
    """Unit test: Test the construction of VectorRangeQuery with various parameters."""
    # Basic range query
    basic_query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score"],
        distance_threshold=0.2,
    )

    query_string = str(basic_query)
    assert "VECTOR_RANGE $distance_threshold $vector" in query_string
    assert "$YIELD_DISTANCE_AS: vector_distance" in query_string
    assert "HYBRID_POLICY" not in query_string

    # Range query with epsilon
    epsilon_query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score"],
        distance_threshold=0.2,
        epsilon=0.05,
    )

    query_string = str(epsilon_query)
    assert "VECTOR_RANGE $distance_threshold $vector" in query_string
    assert "$YIELD_DISTANCE_AS: vector_distance" in query_string
    assert "$EPSILON: 0.05" in query_string
    assert epsilon_query.epsilon == 0.05
    assert "EPSILON" not in epsilon_query.params

    # Range query with hybrid policy
    hybrid_query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score"],
        distance_threshold=0.2,
        hybrid_policy="BATCHES",
    )

    query_string = str(hybrid_query)
    # Hybrid policy should not be in the query string
    assert "HYBRID_POLICY" not in query_string
    assert hybrid_query.hybrid_policy == "BATCHES"
    assert hybrid_query.params["HYBRID_POLICY"] == "BATCHES"

    # Range query with hybrid policy and batch size
    batch_query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        return_fields=["user", "credit_score"],
        distance_threshold=0.2,
        hybrid_policy="BATCHES",
        batch_size=50,
    )

    query_string = str(batch_query)
    # Hybrid policy and batch size should not be in the query string
    assert "HYBRID_POLICY" not in query_string
    assert "BATCH_SIZE" not in query_string
    assert batch_query.hybrid_policy == "BATCHES"
    assert batch_query.batch_size == 50
    assert batch_query.params["HYBRID_POLICY"] == "BATCHES"
    assert batch_query.params["BATCH_SIZE"] == 50


def test_vector_range_query_setter_methods():
    """Unit test: Test setter methods for VectorRangeQuery parameters."""
    # Create a basic query
    query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        distance_threshold=0.2,
    )

    # Verify initial state
    assert query.epsilon is None
    assert query.hybrid_policy is None
    assert query.batch_size is None
    assert "$EPSILON" not in str(query)
    assert "HYBRID_POLICY" not in query.params
    assert "BATCH_SIZE" not in query.params

    # Set epsilon
    query.set_epsilon(0.1)
    assert query.epsilon == 0.1
    assert "$EPSILON: 0.1" in str(query)

    # Set hybrid policy
    query.set_hybrid_policy("BATCHES")
    assert query.hybrid_policy == "BATCHES"
    assert query.params["HYBRID_POLICY"] == "BATCHES"

    # Set batch size
    query.set_batch_size(25)
    assert query.batch_size == 25
    assert query.params["BATCH_SIZE"] == 25


def test_vector_range_query_error_handling():
    """Unit test: Test error handling for invalid VectorRangeQuery parameters."""
    # Create a basic query
    query = VectorRangeQuery(
        vector=[0.1, 0.1, 0.5],
        vector_field_name="user_embedding",
        distance_threshold=0.2,
    )

    # Test invalid epsilon
    with pytest.raises(TypeError, match="epsilon must be of type float or int"):
        query.set_epsilon("0.1")

    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        query.set_epsilon(-0.1)

    # Test invalid hybrid policy
    with pytest.raises(ValueError, match="hybrid_policy must be one of"):
        query.set_hybrid_policy("INVALID")

    # Test invalid batch size
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        query.set_batch_size(10.5)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        query.set_batch_size(0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        query.set_batch_size(-10)
