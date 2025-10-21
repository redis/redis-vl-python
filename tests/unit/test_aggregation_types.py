import pytest
from redis.commands.search.aggregation import AggregateRequest
from redis.commands.search.query import Query
from redis.commands.search.result import Result

from redisvl.index.index import process_results
from redisvl.query.aggregate import HybridQuery, MultiVectorQuery, Vector
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


def test_hybrid_query_text_weights():
    # verify word weights get added into the raw Redis query syntax
    vector = [0.1, 0.1, 0.5]
    vector_field = "user_embedding"

    query = HybridQuery(
        text="query string alpha bravo delta tango alpha",
        text_field_name="description",
        vector=vector,
        vector_field_name=vector_field,
        text_weights={"alpha": 2, "delta": 0.555, "gamma": 0.95},
    )

    assert (
        str(query)
        == "(~@description:(query | string | alpha=>{$weight:2} | bravo | delta=>{$weight:0.555} | tango | alpha=>{$weight:2}))=>[KNN 10 @user_embedding $vector AS vector_distance] SCORER BM25STD ADDSCORES DIALECT 2 APPLY (2 - @vector_distance)/2 AS vector_similarity APPLY @__score AS text_score APPLY 0.30000000000000004*@text_score + 0.7*@vector_similarity AS hybrid_score SORTBY 2 @hybrid_score DESC MAX 10"
    )

    # raise an error if weights are not positive floats
    with pytest.raises(ValueError):
        _ = HybridQuery(
            text="sample text query",
            text_field_name="description",
            vector=vector,
            vector_field_name=vector_field,
            text_weights={"first": 0.2, "second": -0.1},
        )

    with pytest.raises(ValueError):
        _ = HybridQuery(
            text="sample text query",
            text_field_name="description",
            vector=vector,
            vector_field_name=vector_field,
            text_weights={"first": 0.2, "second": "0.1"},
        )

    # no error is weights dictiionary is empty or None
    query = HybridQuery(
        text="sample text query",
        text_field_name="description",
        vector=vector,
        vector_field_name=vector_field,
        text_weights={},
    )
    assert query

    query = HybridQuery(
        text="sample text query",
        text_field_name="description",
        vector=vector,
        vector_field_name=vector_field,
        text_weights=None,
    )
    assert query

    # no error if the words in weights dictionary don't appear in query
    query = HybridQuery(
        text="sample text query",
        text_field_name="description",
        vector=vector,
        vector_field_name=vector_field,
        text_weights={"alpha": 0.2, "bravo": 0.4},
    )
    assert query

    # we can access the word weights on a query object
    assert query.text_weights == {"alpha": 0.2, "bravo": 0.4}

    # we can change the text weights on a query object
    query.set_text_weights(weights={"new": 0.3, "words": 0.125, "here": 99})
    assert query.text_weights == {"new": 0.3, "words": 0.125, "here": 99}

    query.set_text_weights(weights={})
    assert query.text_weights == {}


def test_multi_vector_query():
    # test we require Vector objects
    with pytest.raises(TypeError):
        _ = MultiVectorQuery()

    with pytest.raises(TypeError):
        _ = MultiVectorQuery(vector=[sample_vector])

    with pytest.raises(TypeError):
        _ = MultiVectorQuery(vectors=[[0.1, 0.1, 0.1], "field_1"])

    # test we can initialize with a single vector and single field name
    multivector_query = MultiVectorQuery(
        Vector(vector=sample_vector, field_name="field_1")
    )

    # check default properties
    assert multivector_query._vectors == [
        Vector(vector=sample_vector, field_name="field_1")
    ]
    assert multivector_query._vectors[0].field_name == "field_1"
    assert multivector_query._vectors[0].weight == 1.0
    assert multivector_query._vectors[0].dtype == "float32"
    assert multivector_query._filter_expression == None
    assert multivector_query._num_results == 10
    assert multivector_query._loadfields == []
    assert multivector_query._dialect == 2

    # test we can initialize with multiple Vectors
    vectors = [sample_vector, sample_vector_2, sample_vector_3, sample_vector_4]
    vector_field_names = ["field_1", "field_2", "field_3", "field_4"]
    weights = [0.2, 0.5, 0.6, 0.1]
    dtypes = ["float32", "float32", "float32", "float32"]

    args = []
    for vec, field, weight, dtype in zip(vectors, vector_field_names, weights, dtypes):
        args.append(Vector(vector=vec, field_name=field, weight=weight, dtype=dtype))

    multivector_query = MultiVectorQuery(vectors=args)

    assert len(multivector_query._vectors) == 4
    assert multivector_query._vectors == args

    # test defaults can be overwritten
    filter_expression = Tag("user group") == ["group A", "group C"]

    multivector_query = MultiVectorQuery(
        vectors=args,
        filter_expression=filter_expression,
        num_results=5,
        return_fields=["field_1", "user name", "address"],
        dialect=4,
    )

    assert multivector_query._filter_expression == filter_expression
    assert multivector_query._num_results == 5
    assert multivector_query._loadfields == ["field_1", "user name", "address"]
    assert multivector_query._dialect == 4


def test_multi_vector_query_string():
    # if a single weight is passed it is applied to all similarity scores
    field_1 = "text embedding"
    field_2 = "image embedding"
    weight_1 = 0.2
    weight_2 = 0.7
    multi_vector_query = MultiVectorQuery(
        vectors=[
            Vector(vector=sample_vector_2, field_name=field_1, weight=weight_1),
            Vector(vector=sample_vector_3, field_name=field_2, weight=weight_2),
        ]
    )

    assert (
        str(multi_vector_query)
        == f"@{field_1}:[VECTOR_RANGE 2.0 $vector_0]=>{{$YIELD_DISTANCE_AS: distance_0}} | @{field_2}:[VECTOR_RANGE 2.0 $vector_1]=>{{$YIELD_DISTANCE_AS: distance_1}} SCORER TFIDF DIALECT 2 APPLY (2 - @distance_0)/2 AS score_0 APPLY (2 - @distance_1)/2 AS score_1 APPLY @score_0 * {weight_1} + @score_1 * {weight_2} AS combined_score SORTBY 2 @combined_score DESC MAX 10"
    )


def test_vector_object_validation():
    # test an error is raised if none of the field names are present
    with pytest.raises(ValueError):
        _ = Vector()

    with pytest.raises(ValueError):
        _ = Vector(
            vector=[],
            field_name=[],
        )

    # test an error is raised if the type of vector or fields are incorrect
    # no list of list of floats
    with pytest.raises(ValueError):
        _ = Vector(
            vector=[sample_vector, sample_vector_2, sample_vector_3],
            field_name="text embedding",
        )

    # no list as field name
    with pytest.raises(ValueError):
        _ = Vector(
            vector=sample_vector,
            field_name=["text embedding", "image embedding", "features"],
        )

    # dtype must be one of the supported values
    with pytest.raises(ValueError):
        _ = Vector(vector=sample_vector, field_name="text embedding", dtype="float")

    with pytest.raises(ValueError):
        _ = Vector(vector=sample_vector, field_name="text embedding", dtype="normal")

    with pytest.raises(ValueError):
        _ = Vector(vector=sample_vector, field_name="text embedding", dtype="")

    for dtype in ["bfloat16", "float16", "float32", "float64", "int8", "uint8"]:
        vec = Vector(vector=sample_vector, field_name="text embedding", dtype=dtype)
        assert isinstance(vec, Vector)
