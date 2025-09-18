import pytest
from redis.commands.search.aggregation import AggregateRequest
from redis.commands.search.query import Query
from redis.commands.search.result import Result

from redisvl.index.index import process_results
from redisvl.query.aggregate import HybridQuery
from redisvl.query.filter import Tag

# Sample data for testing
sample_vector = [0.1, 0.2, 0.3, 0.4]
sample_text = "the toon squad play basketball against a gang of aliens"


# Test Cases
def test_aggregate_hybrid_query():
    text_field_name = "description"
    vector_field_name = "embedding"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name=text_field_name,
        vector=sample_vector,
        vector_field_name=vector_field_name,
    )

    assert isinstance(hybrid_query, AggregateRequest)

    # Check default properties
    assert hybrid_query._text == sample_text
    assert hybrid_query._text_field == text_field_name
    assert hybrid_query._vector == sample_vector
    assert hybrid_query._vector_field == vector_field_name
    assert hybrid_query._scorer == "BM25STD"
    assert hybrid_query._filter_expression == None
    assert hybrid_query._alpha == 0.7
    assert hybrid_query._num_results == 10
    assert hybrid_query._loadfields == []
    assert hybrid_query._dialect == 2

    # Check specifying properties
    scorer = "TFIDF"
    filter_expression = Tag("genre") == "comedy"
    alpha = 0.5
    num_results = 8
    return_fields = ["title", "genre", "rating"]
    stopwords = []
    dialect = 2

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name=text_field_name,
        vector=sample_vector,
        vector_field_name=vector_field_name,
        text_scorer=scorer,
        filter_expression=filter_expression,
        alpha=alpha,
        num_results=num_results,
        return_fields=return_fields,
        stopwords=stopwords,
        dialect=dialect,
    )

    assert hybrid_query._text == sample_text
    assert hybrid_query._text_field == text_field_name
    assert hybrid_query._vector == sample_vector
    assert hybrid_query._vector_field == vector_field_name
    assert hybrid_query._scorer == scorer
    assert hybrid_query._filter_expression == filter_expression
    assert hybrid_query._alpha == 0.5
    assert hybrid_query._num_results == 8
    assert hybrid_query._loadfields == return_fields
    assert hybrid_query._dialect == 2
    assert hybrid_query.stopwords == set()

    # Test stopwords are configurable
    hybrid_query = HybridQuery(
        sample_text, text_field_name, sample_vector, vector_field_name, stopwords=None
    )
    assert hybrid_query.stopwords == set([])

    hybrid_query = HybridQuery(
        sample_text,
        text_field_name,
        sample_vector,
        vector_field_name,
        stopwords=["the", "a", "of"],
    )
    assert hybrid_query.stopwords == set(["the", "a", "of"])
    hybrid_query = HybridQuery(
        sample_text,
        text_field_name,
        sample_vector,
        vector_field_name,
        stopwords="german",
    )
    assert hybrid_query.stopwords != set([])

    with pytest.raises(ValueError):
        hybrid_query = HybridQuery(
            sample_text,
            text_field_name,
            sample_vector,
            vector_field_name,
            stopwords="gibberish",
        )

    with pytest.raises(TypeError):
        hybrid_query = HybridQuery(
            sample_text,
            text_field_name,
            sample_vector,
            vector_field_name,
            stopwords=[1, 2, 3],
        )


def test_hybrid_query_with_string_filter():
    """Test that HybridQuery correctly includes string filter expressions in query string.

    This test ensures that when a string filter expression is passed to HybridQuery,
    it's properly included in the generated query string and not set to empty.
    Regression test for bug where string filters were being ignored.
    """
    text = "search for document 12345"
    text_field_name = "description"
    vector_field_name = "embedding"

    # Test with string filter expression - should include filter in query string
    string_filter = "@category:{tech|science|engineering}"
    hybrid_query = HybridQuery(
        text=text,
        text_field_name=text_field_name,
        vector=sample_vector,
        vector_field_name=vector_field_name,
        filter_expression=string_filter,
    )

    # Check that filter is stored correctly
    print("hybrid_query.filter ===", hybrid_query.filter)
    assert hybrid_query._filter_expression == string_filter

    # Check that the generated query string includes both text search and filter
    query_string = str(hybrid_query)
    assert f"@{text_field_name}:(search | document | 12345)" in query_string
    assert f"AND {string_filter}" in query_string

    # Test with FilterExpression - should also work (existing functionality)
    filter_expression = Tag("category") == "tech"
    hybrid_query_with_filter_expr = HybridQuery(
        text=text,
        text_field_name=text_field_name,
        vector=sample_vector,
        vector_field_name=vector_field_name,
        filter_expression=filter_expression,
    )

    # Check that filter is stored correctly
    assert hybrid_query_with_filter_expr._filter_expression == filter_expression

    # Check that the generated query string includes both text search and filter
    query_string_with_filter_expr = str(hybrid_query_with_filter_expr)
    assert (
        f"@{text_field_name}:(search | document | 12345)"
        in query_string_with_filter_expr
    )
    assert "AND @category:{tech}" in query_string_with_filter_expr

    # Test with no filter - should only have text search
    hybrid_query_no_filter = HybridQuery(
        text=text,
        text_field_name=text_field_name,
        vector=sample_vector,
        vector_field_name=vector_field_name,
    )

    query_string_no_filter = str(hybrid_query_no_filter)
    assert f"@{text_field_name}:(search | document | 12345)" in query_string_no_filter
    assert "AND" not in query_string_no_filter

    # Test with wildcard filter - should only have text search (no AND clause)
    hybrid_query_wildcard = HybridQuery(
        text=text,
        text_field_name=text_field_name,
        vector=sample_vector,
        vector_field_name=vector_field_name,
        filter_expression="*",
    )

    query_string_wildcard = str(hybrid_query_wildcard)
    assert f"@{text_field_name}:(search | document | 12345)" in query_string_wildcard
    assert "AND" not in query_string_wildcard
