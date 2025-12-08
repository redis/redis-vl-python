"""Unit tests for HybridQuery class from redisvl.query.hybrid module.

This test module validates the functionality of the HybridQuery class which
combines full-text search with vector similarity search using Redis's hybrid
query capabilities (requires redis>=7.1.0).
"""

from typing import List, Literal

import pytest

from redisvl.redis.utils import array_to_buffer

try:
    from redis.commands.search.hybrid_query import HybridQuery as RedisHybridQuery
    from redis.commands.search.hybrid_query import (
        HybridSearchQuery,
        HybridVsimQuery,
        VectorSearchMethods,
    )

    from redisvl.query.hybrid import HybridQuery, build_combination_method

    REDIS_HYBRID_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    REDIS_HYBRID_AVAILABLE = False
    # Create dummy classes to avoid import errors
    RedisHybridQuery = None  # type: ignore
    HybridSearchQuery = None  # type: ignore
    HybridVsimQuery = None  # type: ignore
    VectorSearchMethods = None  # type: ignore
    HybridQuery = None  # type: ignore

from redisvl.query.filter import Num, Tag, Text

# Test data
sample_vector = [0.1, 0.2, 0.3, 0.4]
bytes_vector = array_to_buffer(sample_vector, "float32")
sample_text = "the toon squad play basketball against a gang of aliens"


def get_query_pieces(query: HybridQuery) -> List[str]:
    """Get all the pieces of the complete hybrid query."""
    # NOTE: Modeled after logic in `redis.commands.search.commands.SearchCommands.hybrid_search`
    pieces = query.query.get_args()
    if query.combination_method:
        pieces.extend(query.combination_method.get_args())
    if query.postprocessing_config.build_args():
        pieces.extend(query.postprocessing_config.build_args())
    return pieces


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

    # Verify get_args() returns empty list (HybridQuery uses params, not args)
    assert get_query_pieces(hybrid_query) == [
        "SEARCH",
        "(~@description:(toon | squad | play | basketball | gang | aliens))",
        "SCORER",
        "BM25STD",
        "VSIM",
        "@embedding",
        bytes_vector,
    ]

    # Verify that no combination method is set
    assert hybrid_query.combination_method is None


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_all_parameters():
    """Test HybridQuery initialization with all optional parameters."""
    filter_expression = Tag("genre") == "comedy"
    text_weights = {"toon": 2.0, "squad": 1.5}

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        text_scorer="TFIDF",
        text_filter_expression=filter_expression,
        yield_text_score_as="text_score",
        vector_search_method="KNN",
        knn_k=10,
        knn_ef_runtime=100,
        yield_vsim_score_as="vsim_score",
        stopwords=None,
        text_weights=text_weights,
        combination_method="RRF",
        rrf_window=10,
        rrf_constant=0.5,
        yield_combined_score_as="hybrid_score",
    )

    assert hybrid_query._ft_helper is not None
    assert hybrid_query._ft_helper.stopwords == set()
    assert hybrid_query._ft_helper.text_weights == text_weights

    # Verify that the expected query pieces have been defined
    assert get_query_pieces(hybrid_query) == [
        "SEARCH",
        "(~@description:(the | toon=>{$weight:2.0} | squad=>{$weight:1.5} | play | basketball | against | a | gang | of | aliens) AND @genre:{comedy})",
        "SCORER",
        "TFIDF",
        "YIELD_SCORE_AS",
        "text_score",
        "VSIM",
        "@embedding",
        bytes_vector,
        "KNN",
        4,
        "K",
        10,
        "EF_RUNTIME",
        100,
        "YIELD_SCORE_AS",
        "vsim_score",
        "COMBINE",
        "RRF",
        6,
        "WINDOW",
        10,
        "CONSTANT",
        0.5,
        "YIELD_SCORE_AS",
        "hybrid_score",
    ]

    # Add post-processing and verify that it is reflected in the query
    hybrid_query.postprocessing_config.limit(offset=10, num=20)
    assert get_query_pieces(hybrid_query)[-3:] == ["LIMIT", "10", "20"]


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
        text_filter_expression=string_filter,
    )

    assert get_query_pieces(hybrid_query) == [
        "SEARCH",
        "(~@description:(toon | squad | play | basketball | gang | aliens) AND @category:{tech|science|engineering})",
        "SCORER",
        "BM25STD",
        "VSIM",
        "@embedding",
        bytes_vector,
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
        text_filter_expression=tag_filter,
    )

    assert get_query_pieces(hybrid_query) == [
        "SEARCH",
        "(~@description:(toon | squad | play | basketball | gang | aliens) AND @genre:{comedy})",
        "SCORER",
        "BM25STD",
        "VSIM",
        "@embedding",
        bytes_vector,
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
        text_filter_expression=numeric_filter,
    )

    # Verify filter is included in serialized query
    args = get_query_pieces(hybrid_query)
    assert args[1].endswith("AND @age:[(30 +inf])")


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_text_filter():
    """Test HybridQuery with Text FilterExpression."""
    text_filter = Text("job") == "engineer"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        text_filter_expression=text_filter,
    )

    # Verify filter is included in serialized query
    args = get_query_pieces(hybrid_query)
    assert args[1].endswith('AND @job:("engineer"))')


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_combined_filters():
    """Test HybridQuery with combined FilterExpressions."""
    combined_filter = (Tag("genre") == "comedy") & (Num("rating") > 7.0)

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        text_filter_expression=combined_filter,
    )

    # Verify both filters are included in serialized query
    args = get_query_pieces(hybrid_query)
    assert args[1].endswith("AND (@genre:{comedy} @rating:[(7.0 +inf]))")


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_wildcard_filter():
    """Test HybridQuery with wildcard filter."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        text_filter_expression="*",
    )

    # Verify query structure - wildcard may or may not be included depending on implementation
    args = get_query_pieces(hybrid_query)
    assert (
        args[1] == "(~@description:(toon | squad | play | basketball | gang | aliens))"
    )  # Query without filtering


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_without_filter():
    """Test HybridQuery without any filter expression."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        text_filter_expression=None,
    )

    # Verify no filter in serialized query (only text query)
    args = get_query_pieces(hybrid_query)
    assert (
        args[1] == "(~@description:(toon | squad | play | basketball | gang | aliens))"
    )  # No filter in query


# Vector search method tests


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_vector_search_method_knn():
    """Test HybridQuery with KNN vector search method."""
    with pytest.raises(ValueError):
        # KNN requires K
        HybridQuery(
            text=sample_text,
            text_field_name="description",
            vector=sample_vector,
            vector_field_name="embedding",
            vector_search_method="KNN",
        )

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method="KNN",
        knn_k=10,
    )

    # KNN with params should be in args
    args = get_query_pieces(hybrid_query)
    assert args[-4:] == ["KNN", 2, "K", 10]

    # With optional EF_RUNTIME param
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method="KNN",
        knn_k=10,
        knn_ef_runtime=100,
    )

    # KNN with params should be in args
    args = get_query_pieces(hybrid_query)
    assert args[-6:] == ["KNN", 4, "K", 10, "EF_RUNTIME", 100]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_vector_search_method_range():
    """Test HybridQuery with RANGE vector search method."""
    with pytest.raises(ValueError):
        # RANGE requires RADIUS
        HybridQuery(
            text=sample_text,
            text_field_name="description",
            vector=sample_vector,
            vector_field_name="embedding",
            vector_search_method="RANGE",
        )

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method="RANGE",
        range_radius=10,
    )

    # RANGE with params should be in args
    args = get_query_pieces(hybrid_query)
    assert args[-4:] == ["RANGE", 2, "RADIUS", 10]

    # With optional EPSILON param
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method="RANGE",
        range_radius=10,
        range_epsilon=0.1,
    )

    # RANGE with params should be in args
    args = get_query_pieces(hybrid_query)
    assert args[-6:] == ["RANGE", 4, "RADIUS", 10, "EPSILON", 0.1]


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
    args = get_query_pieces(hybrid_query)
    assert "VSIM" in args
    assert "@embedding" in args
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

    assert get_query_pieces(hybrid_query) == [
        "SEARCH",
        "(~@description:(search | \\@user | \\#hashtag | \\$price | 50\\% | off\\!))",
        "SCORER",
        "BM25STD",
        "VSIM",
        "@embedding",
        bytes_vector,
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

    assert get_query_pieces(hybrid_query) == [
        "SEARCH",
        "(~@description:(café | résumé | naïve | 日本語 | 中文))",
        "SCORER",
        "BM25STD",
        "VSIM",
        "@embedding",
        bytes_vector,
    ]


# Vector filter expression tests


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_vector_filter_tag():
    """Test HybridQuery with Tag FilterExpression on vector search."""
    tag_filter = Tag("genre") == "comedy"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_filter_expression=tag_filter,
    )

    # Verify filter is included in serialized query
    args = get_query_pieces(hybrid_query)
    assert args[-2:] == ["FILTER", "@genre:{comedy}"]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_vector_filter_string():
    """Test HybridQuery with string filter expression on vector search."""
    string_filter = "@category:{tech|science|engineering}"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_filter_expression=string_filter,
    )

    # Verify filter is included in serialized query
    args = get_query_pieces(hybrid_query)
    assert args[-2:] == ["FILTER", "@category:{tech|science|engineering}"]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_vector_filter_numeric():
    """Test HybridQuery with Numeric FilterExpression on vector search."""
    numeric_filter = Num("rating") > 7.0

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_filter_expression=numeric_filter,
    )

    # Verify filter is included in serialized query
    args = get_query_pieces(hybrid_query)
    assert args[-2:] == ["FILTER", "@rating:[(7.0 +inf]"]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_vector_filter_text():
    """Test HybridQuery with Text FilterExpression on vector search."""
    text_filter = Text("job") == "engineer"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_filter_expression=text_filter,
    )

    # Verify filter is included in serialized query
    args = get_query_pieces(hybrid_query)
    assert args[-2:] == ["FILTER", '@job:("engineer")']


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_vector_filter_combined():
    """Test HybridQuery with combined FilterExpressions on vector search."""
    combined_filter = (Tag("genre") == "comedy") & (Num("rating") > 7.0)

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_filter_expression=combined_filter,
    )

    # Verify both filters are included in serialized query
    args = get_query_pieces(hybrid_query)
    assert args[-2:] == ["FILTER", "(@genre:{comedy} @rating:[(7.0 +inf])"]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_vector_filter_none():
    """Test HybridQuery without vector filter expression."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_filter_expression=None,
    )

    # Verify no FILTER in serialized query
    args = get_query_pieces(hybrid_query)
    assert "FILTER" not in args


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_vector_filter_and_method():
    """Test HybridQuery with vector filter and a search method."""
    tag_filter = Tag("genre") == "comedy"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        vector_search_method="KNN",
        knn_k=10,
        vector_filter_expression=tag_filter,
    )

    # Verify KNN params and filter are both in args
    args = get_query_pieces(hybrid_query)
    assert args[-6:] == ["KNN", 2, "K", 10, "FILTER", "@genre:{comedy}"]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_with_both_text_and_vector_filters():
    """Test HybridQuery with both text_filter_expression and vector_filter_expression."""
    text_filter = Tag("category") == "movies"
    vector_filter = Tag("genre") == "comedy"

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        text_filter_expression=text_filter,
        vector_filter_expression=vector_filter,
    )

    # Verify both filters are in the query
    args = get_query_pieces(hybrid_query)
    assert args == [
        "SEARCH",
        "(~@description:(toon | squad | play | basketball | gang | aliens) AND @category:{movies})",
        "SCORER",
        "BM25STD",
        "VSIM",
        "@embedding",
        bytes_vector,
        "FILTER",
        "@genre:{comedy}",
    ]


# Combination method tests


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_combination_method_rrf_basic():
    """Test HybridQuery with RRF combination method."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        combination_method="RRF",
        rrf_window=10,
    )

    # Verify RRF combination method is set
    assert hybrid_query.combination_method is not None

    # Verify that combination method args are correct
    assert hybrid_query.combination_method.get_args() == [
        "COMBINE",
        "RRF",
        2,
        "WINDOW",
        10,
    ]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_combination_method_rrf_with_constant():
    """Test HybridQuery with RRF combination method and constant parameter."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        combination_method="RRF",
        rrf_constant=0.5,
    )

    # Verify RRF combination method is set
    assert hybrid_query.combination_method is not None

    # Verify that combination method args are correct
    assert hybrid_query.combination_method.get_args() == [
        "COMBINE",
        "RRF",
        2,
        "CONSTANT",
        0.5,
    ]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_combination_method_rrf_with_both_params():
    """Test HybridQuery with RRF combination method with both window and constant."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        combination_method="RRF",
        rrf_window=20,
        rrf_constant=1.0,
        yield_combined_score_as="rrf_score",
    )

    # Verify RRF combination method is set
    assert hybrid_query.combination_method is not None

    # Verify that combination method args are correct
    assert hybrid_query.combination_method.get_args() == [
        "COMBINE",
        "RRF",
        6,
        "WINDOW",
        20,
        "CONSTANT",
        1.0,
        "YIELD_SCORE_AS",
        "rrf_score",
    ]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_hybrid_query_combination_method_linear_with_alpha(alpha: float):
    """Test HybridQuery with LINEAR combination method."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        combination_method="LINEAR",
        linear_alpha=alpha,
    )

    # Verify LINEAR combination method is set
    assert hybrid_query.combination_method is not None

    # Verify that combination method args are correct
    assert hybrid_query.combination_method.get_args() == [
        "COMBINE",
        "LINEAR",
        4,
        "ALPHA",
        alpha,
        "BETA",
        1 - alpha,
    ]

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        combination_method="LINEAR",
        linear_beta=alpha,
    )

    # Verify LINEAR combination method is set
    assert hybrid_query.combination_method is not None

    # Verify that combination method args are correct
    assert hybrid_query.combination_method.get_args() == [
        "COMBINE",
        "LINEAR",
        4,
        "ALPHA",
        1 - alpha,
        "BETA",
        alpha,
    ]

    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        combination_method="LINEAR",
        linear_alpha=alpha,
        linear_beta=2 * alpha,
    )

    # Verify LINEAR combination method is set
    assert hybrid_query.combination_method is not None

    # Verify that combination method args are correct
    assert hybrid_query.combination_method.get_args() == [
        "COMBINE",
        "LINEAR",
        4,
        "ALPHA",
        alpha,
        "BETA",
        2 * alpha,
    ]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_combination_method_linear_with_yield_score():
    """Test HybridQuery with LINEAR combination method and yield_combined_score_as."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        combination_method="LINEAR",
        linear_alpha=0.3,
        linear_beta=0.7,
        yield_combined_score_as="linear_score",
    )

    assert hybrid_query.combination_method is not None
    assert hybrid_query.combination_method.get_args() == [
        "COMBINE",
        "LINEAR",
        6,
        "ALPHA",
        0.3,
        "BETA",
        0.7,
        "YIELD_SCORE_AS",
        "linear_score",
    ]


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_hybrid_query_combination_method_none():
    """Test HybridQuery without combination method."""
    hybrid_query = HybridQuery(
        text=sample_text,
        text_field_name="description",
        vector=sample_vector,
        vector_field_name="embedding",
        combination_method=None,
    )

    # Verify no combination method is set
    assert hybrid_query.combination_method is None

    # Verify COMBINE does not appear in query args
    args = get_query_pieces(hybrid_query)
    assert "COMBINE" not in args


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
def test_build_combination_method_invalid_method():
    """Test build_combination_method static method with invalid combination method."""
    with pytest.raises(ValueError, match="Unknown combination method"):
        build_combination_method(
            combination_method="INVALID",  # type: ignore
        )


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis>=7.1.0")
@pytest.mark.parametrize("method", ["RRF", "LINEAR"])
def test_build_combination_method_no_parameters(method: Literal["RRF", "LINEAR"]):
    """Test build_combination_method static method raises ValueError when no parameters provided."""
    with pytest.raises(
        ValueError,
        match="No parameters provided for combination method - must provide at least one parameter",
    ):
        build_combination_method(
            combination_method=method,
        )
