"""Unit tests for FullTextQueryHelper tokenization."""

from redisvl.query.aggregate import AggregateHybridQuery
from redisvl.utils.full_text_query_helper import FullTextQueryHelper


def test_stopword_with_apostrophe_is_filtered():
    """Stopword membership is checked against the word the user typed."""
    helper = FullTextQueryHelper()
    assert helper._tokenize_and_escape_query("it's the fox") == "fox"


def test_text_weight_key_with_apostrophe_applies():
    """text_weights keys match the raw word, not its escaped form."""
    helper = FullTextQueryHelper(stopwords=None, text_weights={"don't": 2.0})
    assert helper._tokenize_and_escape_query("don't jump") == (
        "don\\'t=>{$weight:2.0} | jump"
    )


def test_aggregate_hybrid_text_clause_drops_apostrophe_stopword():
    """The clause a hybrid query sends contains no escaped stopword."""
    query = AggregateHybridQuery(
        text="it's the fox",
        text_field_name="body",
        vector=[1.0, 0.0],
        vector_field_name="v",
    )
    assert "(~@body:(fox))" in str(query)
