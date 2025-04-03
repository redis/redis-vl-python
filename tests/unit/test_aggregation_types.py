import pytest
from redis.commands.search.aggregation import AggregateRequest
from redis.commands.search.query import Query
from redis.commands.search.result import Result

from redisvl.index.index import process_results
from redisvl.query.aggregate import HybridAggregationQuery
from redisvl.query.filter import Tag

# Sample data for testing
sample_vector = [0.1, 0.2, 0.3, 0.4]
sample_text = "the toon squad play basketball against a gang of aliens"


# Test Cases
def test_aggregate_hybrid_query():
    text_field_name = "description"
    vector_field_name = "embedding"

    hybrid_query = HybridAggregationQuery(
        text=sample_text,
        text_field_name=text_field_name,
        vector=sample_vector,
        vector_field_name=vector_field_name,
    )

    assert isinstance(hybrid_query, AggregateRequest)

    # Check defaut properties
    assert hybrid_query._text == sample_text
    assert hybrid_query._text_field == text_field_name
    assert hybrid_query._vector == sample_vector
    assert hybrid_query._vector_field == vector_field_name
    assert hybrid_query._scorer == "BM25STD"
    assert hybrid_query._filter_expression == None
    assert hybrid_query._alpha == 0.7
    assert hybrid_query._num_results == 10
    assert hybrid_query._loadfields == []
    assert hybrid_query._dialect == 4

    # Check specifying properties
    scorer = "TFIDF"
    filter_expression = Tag("genre") == "comedy"
    alpha = 0.5
    num_results = 8
    return_fields = ["title", "genre", "rating"]
    stopwords = []
    dialect = 2

    hybrid_query = HybridAggregationQuery(
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
    hybrid_query = HybridAggregationQuery(
        sample_text, text_field_name, sample_vector, vector_field_name, stopwords=None
    )
    assert hybrid_query.stopwords == set([])

    hybrid_query = HybridAggregationQuery(
        sample_text,
        text_field_name,
        sample_vector,
        vector_field_name,
        stopwords=["the", "a", "of"],
    )
    assert hybrid_query.stopwords == set(["the", "a", "of"])
    hybrid_query = HybridAggregationQuery(
        sample_text,
        text_field_name,
        sample_vector,
        vector_field_name,
        stopwords="german",
    )
    assert hybrid_query.stopwords != set([])

    with pytest.raises(ValueError):
        hybrid_query = HybridAggregationQuery(
            sample_text,
            text_field_name,
            sample_vector,
            vector_field_name,
            stopwords="gibberish",
        )

    with pytest.raises(TypeError):
        hybrid_query = HybridAggregationQuery(
            sample_text,
            text_field_name,
            sample_vector,
            vector_field_name,
            stopwords=[1, 2, 3],
        )
