import pytest
from redis.commands.search.aggregation import AggregateRequest
from redis.commands.search.query import Query
from redis.commands.search.result import Result

from redisvl.index.index import process_results
from redisvl.query.aggregate import HybridQuery, MultiVectorQuery
from redisvl.query.filter import Tag

# Sample data for testing
sample_vector = [0.1, 0.2, 0.3, 0.4]
sample_text = "the toon squad play basketball against a gang of aliens"

sample_vector_2 = [0.1, 0.2, 0.3, 0.4]
sample_vector_3 = [0.5, 0.5]
sample_vector_4 = [0.1, 0.1, 0.1]


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


def test_aggregate_multi_vector_query():
    # test we require vectors and field names
    with pytest.raises(ValueError):
        _ = MultiVectorQuery()

    with pytest.raises(ValueError):
        _ = MultiVectorQuery(vectors=[sample_vector], vector_field_names=[])

    with pytest.raises(ValueError):
        _ = MultiVectorQuery(vectors=[], vector_field_names=["field 1"])

    # test we can initialize with a single vector and single field name
    multivector_query = MultiVectorQuery(
        vectors=[sample_vector], vector_field_names=["field 1"]
    )
    assert query.query is not None

    # check default properties
    assert multivector_query._vectors == [sample_vector]
    assert multivector_query._vector_field_names == ["field 1"]
    assert multivector_query._filter_expression == None
    assert multivector_query._weights == 1.0
    assert multivector_query._num_results == 10
    assert multivector_query._loadfields == []
    assert multivector_query._dialect == 2

    # test we can initialize with mutliple vectors and field names
    multivector_query = MultiVectorQuery(
        vectors=[sample_vector, sample_vector_2, sample_vector_3, sample_vector_4],
        vector_field_names=["field 1", "field 2", "field 3", "field 4"],
        weights=[0.2, 0.5, 0.6, 0, 1],
        dtypes=[],
    )

    assert len(multivector_query._vectors) == 4
    assert len(multivector_query._vector_field_names) == 4
    assert len(multivector_query._weights) == 4

    # test defaults can be overwritten
    multivector_query = MultiVectorQuery(
        vectors=[sample_vector, sample_vector_2, sample_vector_3, sample_vector_4],
        vector_field_names=["field 1", "field 2", "field 3", "field 4"],
        filter_expression=(Tag("user group") == ["group A", "group C"]),
        weights=[0.2, 0.5, 0.6, 0, 1],
        dtypes=["float32", "float32", "float64", "bfloat16"],
        num_results=5,
        return_fields=["field 1", "user name", "address"],
        dialect=4,
    )

    assert multivector_query._vectors == [
        sample_vector,
        sample_vector_2,
        sample_vector_3,
        sample_vector_4,
    ]
    assert multivector_query._vector_field_names == [
        "field 1",
        "field 2",
        "field 3",
        "field 4",
    ]
    assert multivector_query._weights == [0.2, 0.5, 0.6, 0, 1]
    assert multivector_query._filter_expression == Tag("user group")
    assert multivector_query._num_results == 5
    assert multivector_query._loadfields == ["field 1", "user name", "address"]
    assert multivector_query._dialect == 4


def test_aggregate_multi_vector_query_broadcasting():
    # if a single vector and multiple fields is passed we search with the same vector over all fields
    multivector_query = MultiVectorQuery(
        vectors=[sample_vector],
        vector_field_names=["text embedding", "image embedding"],
    )
    assert multi_vector_query.query == "<raw text here>"

    # vector being broadcast doesn't need to be in a list
    multivector_query = MultiVectorQuery(
        vectors=sample_vector, vector_field_names=["text embedding", "image embedding"]
    )
    assert multi_vector_query.query == "<raw text here>"

    # if multiple vectors are passed and a single field name we search with all vectors on that field
    multivector_query = MultiVectorQuery(
        vectors=[sample_vector_2, sample_vector_3],
        vector_field_names=["text embedding"],
    )
    assert multi_vector_query.query == "<raw text here>"

    # vector field name does not need to be in a list if only one is provided
    multivector_query = MultiVectorQuery(
        vectors=[sample_vector_2, sample_vector_3], vector_field_names="text embedding"
    )
    assert multi_vector_query.query == "<raw text here>"

    # if a single weight is passed it is applied to all similarity scores
    multivector_query = MultiVectorQuery(
        vectors=[sample_vector_2, sample_vector_3],
        vector_field_names=["text embedding", "image embedding"],
        weights=[0.2],
    )
    assert multi_vector_query.query == "<raw text here>"

    # weight does not need to be in a list if only one is provided
    multivector_query = MultiVectorQuery(
        vectors=[sample_vector_2, sample_vector_3],
        vector_field_names=["text embedding", "image embedding"],
        weights=0.2,
    )
    assert multi_vector_query.query == "<raw text here>"


def test_aggregate_multi_vector_query_errors():
    # test an error is raised if the number of vectors and number of fields don't match
    with pytest.raises(ValueError):
        _ = MultiVectorQuery(
            vectors=[sample_vector, sample_vector_2, sample_vector_3],
            vector_field_names=["text embedding", "image embedding"],
        )

    with pytest.raises(ValueError):
        _ = MultiVectorQuery(
            vectors=[sample_vector, sample_vector_2],
            vector_field_names=["text embedding", "image embedding", "features"],
        )

    # test an error is raised if the number of weights is incorrect
    with pytest.raises(ValueError):
        _ = MultiVectorQuery(
            vectors=[sample_vector, sample_vector_2],
            vector_field_names=["text embedding", "image embedding"],
            weights=[0.1, 0.2, 0.3],
        )

    # test an error is raised if none of the field names are present
    with pytest.raises(ValueError):
        _ = MultiVectorQuery(
            vectors=[],
            vector_field_names=[],
        )
