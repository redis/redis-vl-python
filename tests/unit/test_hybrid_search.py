"""Unit tests for HybridQuery class from redisvl.query.hybrid module.

This test module validates the functionality of the HybridQuery class which
combines full-text search with vector similarity search using Redis's hybrid
query capabilities (requires redis>=7.1.0).
"""

import pytest

try:
    from redis.commands.search.hybrid_query import HybridQuery as _HybridQuery
    from redis.commands.search.hybrid_query import (
        HybridSearchQuery,
        HybridVsimQuery,
        VectorSearchMethods,
    )

    from redisvl.query.hybrid import HybridQuery

    REDIS_HYBRID_AVAILABLE = True
except ImportError:
    REDIS_HYBRID_AVAILABLE = False
    # Create dummy classes to avoid import errors
    _HybridQuery = None  # type: ignore
    HybridSearchQuery = None  # type: ignore
    HybridVsimQuery = None  # type: ignore
    VectorSearchMethods = None  # type: ignore
    HybridQuery = None  # type: ignore

from redisvl.query.filter import Num, Tag, Text

# Test data
sample_vector = [0.1, 0.2, 0.3, 0.4]
sample_text = "the toon squad play basketball against a gang of aliens"

sample_vector_2 = [0.1, 0.2, 0.3, 0.4]
sample_vector_3 = [0.5, 0.5]
sample_vector_4 = [0.1, 0.1, 0.1]


# Basic init tests


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_basic_initialization():
    """Test basic HybridQuery initialization with required parameters."""
    text_field_name = "description"
    vector_field_name = "embedding"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name=text_field_name,
        vector=sample_vector,
        vector_field_name=vector_field_name,
    )

    # Verify it's an instance of the base Redis HybridQuery
    assert isinstance(hybrid_query, _HybridQuery)

    # Verify the FullTextQueryHelper was created
    assert hasattr(hybrid_query, "_ft_helper")
    assert hybrid_query._ft_helper is not None

    # Verify internal components exist (note: _vector_similarity_query, not _vsim_query)
    assert hasattr(hybrid_query, "_search_query")
    assert hasattr(hybrid_query, "_vector_similarity_query")

    # Verify they are the correct types
    assert isinstance(hybrid_query._search_query, HybridSearchQuery)
    assert isinstance(hybrid_query._vector_similarity_query, HybridVsimQuery)

    # Verify get_args() returns empty list (HybridQuery uses params, not args)
    assert hybrid_query.get_args() == [
        "SEARCH",
        "(~@description:(toon | squad | play | basketball | gang | aliens)",
        "SCORER",
        "BM25STD",
        "VSIM",
        "embedding",
        [0.1, 0.2, 0.3, 0.4],
    ]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_all_parameters():
    """Test HybridQuery initialization with all optional parameters."""
    text_field_name = "description"
    vector_field_name = "embedding"
    text_scorer = "TFIDF"
    filter_expression = Tag("genre") == "comedy"
    vector_search_method = "KNN"
    vector_search_method_params = {"K": 10}
    stopwords = None
    text_weights = {"toon": 2.0, "squad": 1.5}

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name=text_field_name,
        vector=sample_vector,
        vector_field_name=vector_field_name,
        text_scorer=text_scorer,
        filter_expression=filter_expression,
        vector_search_method=vector_search_method,
        vector_search_method_params=vector_search_method_params,
        stopwords=stopwords,
        text_weights=text_weights,
    )

    assert isinstance(hybrid_query, _HybridQuery)
    assert hybrid_query._ft_helper is not None
    assert hybrid_query._ft_helper.stopwords == set()
    assert hybrid_query._ft_helper.text_weights == text_weights

    # Verify get_args() returns correct serialized query with all parameters
    args = hybrid_query.get_args()
    expected = [
        "SEARCH",
        "(~@description:(the | toon=>{$weight:2.0} | squad=>{$weight:1.5} | play | basketball | against | a | gang | of | aliens) AND @genre:{comedy}",
        "SCORER",
        "TFIDF",
        "VSIM",
        "embedding",
        [0.1, 0.2, 0.3, 0.4],
        "KNN",
        2,
        "K",
        10,
    ]
    assert args == expected


# Stopwords tests


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_stopwords_default():
    """Test that default stopwords (english) are applied."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
    )

    # Default should be english stopwords
    stopwords = hybrid_query._ft_helper.stopwords
    assert isinstance(stopwords, set)
    assert len(stopwords) > 0
    # Common english stopwords should be present
    assert "the" in stopwords
    assert "a" in stopwords


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_stopwords_none():
    """Test that stopwords can be disabled with None."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        stopwords=None,
    )

    assert hybrid_query._ft_helper.stopwords == set()


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_stopwords_empty_set():
    """Test that stopwords can be set to empty set."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        stopwords=set(),
    )

    assert hybrid_query._ft_helper.stopwords == set()


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_stopwords_custom():
    """Test that custom stopwords are applied."""
    custom_stopwords = {"the", "a", "of", "and"}

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        stopwords=custom_stopwords,
    )

    assert hybrid_query._ft_helper.stopwords == set(custom_stopwords)


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_stopwords_language():
    """Test that language-specific stopwords can be loaded."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        stopwords="german",
    )

    # German stopwords should be loaded
    stopwords = hybrid_query._ft_helper.stopwords
    assert isinstance(stopwords, set)
    assert len(stopwords) > 0
    for word in ("der", "die", "und"):  # Common expected words
        assert word in stopwords


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_stopwords_invalid_language():
    """Test that invalid language raises ValueError."""
    with pytest.raises(ValueError):
        HybridQuery(
            text=sample_text,
            text_field_name="description",
            vector=sample_vector,
            vector_field_name="embedding",
            stopwords="gibberish_language",
        )


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_stopwords_invalid_type():
    """Test that invalid stopwords type raises TypeError."""
    with pytest.raises(TypeError):
        HybridQuery(
            text=sample_text,
            text_field_name="description",
            vector=sample_vector,
            vector_field_name="embedding",
            stopwords=[1, 2, 3],  # Invalid: list of integers
        )


# Text weight tests


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_text_weights_basic():
    """Test that text weights are properly applied."""
    text_weights = {"toon": 2.0, "squad": 1.5, "basketball": 3.0}

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        text_weights=text_weights,
    )

    assert hybrid_query._ft_helper.text_weights == text_weights


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_text_weights_none():
    """Test that text_weights can be None."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        text_weights=None,
    )

    assert hybrid_query._ft_helper.text_weights == {}


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_text_weights_empty():
    """Test that text_weights can be empty dict."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        text_weights={},
    )

    assert hybrid_query._ft_helper.text_weights == {}


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_text_weights_negative_value():
    """Test that negative text weights raise ValueError."""
    with pytest.raises(ValueError):
        HybridQuery(
            text=sample_text,
            text_field_name="description",
            vector=sample_vector,
            vector_field_name="embedding",
            text_weights={"word": -0.5},
        )


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_text_weights_invalid_type():
    """Test that non-numeric text weights raise ValueError."""
    with pytest.raises(ValueError):
        HybridQuery(
            text=sample_text,
            text_field_name="description",
            vector=sample_vector,
            vector_field_name="embedding",
            text_weights={"word": "invalid"},  # type: ignore
        )


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_text_weights_multi_word_key():
    """Test that multi-word keys in text_weights raise ValueError."""
    with pytest.raises(ValueError):
        HybridQuery(
            text=sample_text,
            text_field_name="description",
            vector=sample_vector,
            vector_field_name="embedding",
            text_weights={"multi word": 2.0},
        )


# Filter expression tests


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_string_filter():
    """Test HybridQuery with string filter expression."""
    string_filter = "@category:{tech|science|engineering}"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        filter_expression=string_filter,
    )

    assert hybrid_query.get_args() == [
        "SEARCH",
        "(~@description:(toon | squad | play | basketball | gang | aliens) AND "
        "@category:{tech|science|engineering}",
        "SCORER",
        "BM25STD",
        "VSIM",
        "embedding",
        [0.1, 0.2, 0.3, 0.4],
    ]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_tag_filter():
    """Test HybridQuery with Tag FilterExpression."""
    tag_filter = Tag("genre") == "comedy"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        filter_expression=tag_filter,
    )

    assert hybrid_query.get_args() == [
        "SEARCH",
        "(~@description:(toon | squad | play | basketball | gang | aliens) AND "
        "@genre:{comedy}",
        "SCORER",
        "BM25STD",
        "VSIM",
        "embedding",
        [0.1, 0.2, 0.3, 0.4],
    ]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_numeric_filter():
    """Test HybridQuery with Numeric FilterExpression."""
    numeric_filter = Num("age") > 30

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        filter_expression=numeric_filter,
    )

    # Verify filter is included in serialized query
    args = hybrid_query.get_args()
    assert args[0] == "SEARCH"
    assert "@age:[(30 +inf]" in args[1]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_text_filter():
    """Test HybridQuery with Text FilterExpression."""
    text_filter = Text("job") == "engineer"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        filter_expression=text_filter,
    )

    # Verify filter is included in serialized query
    args = hybrid_query.get_args()
    assert args[0] == "SEARCH"
    assert '@job:("engineer")' in args[1]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_combined_filters():
    """Test HybridQuery with combined FilterExpressions."""
    combined_filter = (Tag("genre") == "comedy") & (Num("rating") > 7.0)

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        filter_expression=combined_filter,
    )

    # Verify both filters are included in serialized query
    args = hybrid_query.get_args()
    assert args[0] == "SEARCH"
    assert "@genre:{comedy}" in args[1]
    assert "@rating:[(7/- +inf]" in args[1]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_wildcard_filter():
    """Test HybridQuery with wildcard filter."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        filter_expression="*",
    )

    # Verify query structure - wildcard may or may not be included depending on implementation
    args = hybrid_query.get_args()
    assert (
        args[1] == "(~@description:(toon | squad | play | basketball | gang | aliens)"
    )  # Query without filtering


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_without_filter():
    """Test HybridQuery without any filter expression."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        filter_expression=None,
    )

    # Verify no filter in serialized query (only text query)
    args = hybrid_query.get_args()
    assert (
        args[1] == "(~@description:(toon | squad | play | basketball | gang | aliens)"
    )  # No filter in query


# Vector search method tests


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_vector_search_method_knn():
    """Test HybridQuery with KNN vector search method."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method="KNN",
    )

    # Without KNN params, it should not be in the args
    args = hybrid_query.get_args()
    assert "KNN" not in args

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method="KNN",
        vector_search_method_params={"K": 5},
    )

    # KNN with params should be in args
    args = hybrid_query.get_args()
    assert args[-4:] == ["KNN", 2, "K", 5]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_vector_search_method_range():
    """Test HybridQuery with RANGE vector search method."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method="RANGE",
    )

    # Without RANGE params, it should not be in the args
    args = hybrid_query.get_args()
    assert "RANGE" not in args

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method="RANGE",
        vector_search_method_params={"RADIUS": 10},
    )

    # RANGE with params should be in args
    args = hybrid_query.get_args()
    assert args[-4:] == ["RANGE", 2, "RADIUS", 10]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_vector_search_method_none():
    """Test HybridQuery without specifying vector search method."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method=None,
    )

    # Verify basic VSIM structure without explicit method
    args = hybrid_query.get_args()
    assert "VSIM" in args
    assert "embedding" in args
    # When None, should not have KNN or RANGE explicitly
    assert "KNN" not in args
    assert "RANGE" not in args


# Edge cases


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_empty_text_after_stopwords():
    """Test HybridQuery behavior when text becomes empty after stopword removal."""
    # All words are stopwords
    with pytest.raises(ValueError, match="text string cannot be empty"):
        HybridQuery(
            text="the a an",
            text_field_name="description",
            vector=sample_vector,
            vector_field_name="embedding",
            stopwords="english",
        )


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_special_characters_in_text():
    """Test HybridQuery with special characters in text."""
    special_text = "search for @user #hashtag $price 50% off!"

    hybrid_query = HybridQuery(
        text=special_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
    )

    assert hybrid_query.get_args() == [
        "SEARCH",
        "(~@description:(search | \\@user | \\#hashtag | \\$price | 50\\% | off\\!)",
        "SCORER",
        "BM25STD",
        "VSIM",
        "embedding",
        [0.1, 0.2, 0.3, 0.4],
    ]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_unicode_text():
    """Test HybridQuery with Unicode characters in text."""
    unicode_text = "café résumé naïve 日本語 中文"

    hybrid_query = HybridQuery(
        text=unicode_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        stopwords=None,  # Disable stopwords for Unicode test
    )

    assert hybrid_query.get_args() == [
        "SEARCH",
        "(~@description:(café | résumé | naïve | 日本語 | 中文)",
        "SCORER",
        "BM25STD",
        "VSIM",
        "embedding",
        [0.1, 0.2, 0.3, 0.4],
    ]
