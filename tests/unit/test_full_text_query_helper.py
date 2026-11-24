"""Unit tests for FullTextQueryHelper tokenization."""

import pytest

from redisvl.query.aggregate import AggregateHybridQuery
from redisvl.utils.full_text_query_helper import FullTextQueryHelper

try:
    # redisvl.query.hybrid imports on any redis-py; building a HybridQuery
    # needs redis-py>=7.1.0, which provides this module.
    import redis.commands.search.hybrid_query  # noqa: F401

    from redisvl.query.hybrid import HybridQuery
except ImportError:
    HybridQuery = None  # type: ignore


def _hybrid_classes():
    classes = [pytest.param(AggregateHybridQuery, id="AggregateHybridQuery")]
    classes.append(
        pytest.param(
            HybridQuery,
            id="HybridQuery",
            marks=pytest.mark.skipif(
                HybridQuery is None, reason="Requires redis-py>=7.1.0"
            ),
        )
    )
    return classes


def test_stopword_with_apostrophe_is_filtered():
    """Stopword membership is checked against the word the user typed."""
    helper = FullTextQueryHelper()
    assert helper.build_query_string("it's the fox", "body") == "(~@body:(fox))"


def test_text_weight_key_with_apostrophe_is_attached_to_its_term():
    """text_weights keys match the raw word, not its escaped form."""
    helper = FullTextQueryHelper(stopwords=None, text_weights={"don't": 2.0})
    assert helper.build_query_string("don't jump", "body") == (
        "(~@body:(don\\'t=>{$weight:2.0} | jump))"
    )


@pytest.mark.parametrize("query_class", _hybrid_classes())
def test_hybrid_text_clause_drops_apostrophe_stopword(query_class):
    """The clause both hybrid queries send contains no escaped stopword."""
    query = query_class(
        text="it's the fox",
        text_field_name="body",
        vector=[1.0, 0.0],
        vector_field_name="v",
    )
    sent = str(query) if query_class is AggregateHybridQuery else query.query.get_args()
    assert "(~@body:(fox))" in str(sent)


@pytest.mark.parametrize("text", ["it's", "won't", "you're not"])
def test_contraction_only_text_is_rejected(text):
    """Text made only of contractions and stopwords now fails at construction."""
    helper = FullTextQueryHelper()
    with pytest.raises(ValueError, match="empty after removing stopwords"):
        helper.build_query_string(text, "body")
