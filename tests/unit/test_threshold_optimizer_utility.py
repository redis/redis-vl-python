import sys

import pytest

if sys.version_info.major == 3 and sys.version_info.minor < 10:
    pytest.skip("Test requires Python 3.10 or higher", allow_module_level=True)

from ranx import evaluate

from redisvl.extensions.threshold_optimizer.cache import _generate_run_cache
from redisvl.extensions.threshold_optimizer.schema import TestData
from redisvl.extensions.threshold_optimizer.utils import _format_qrels

# Note: these tests are not intended to test ranx but to test that our data formatting for the package is correct


def test_known_precision_case():
    """
    Test case with known precision value.

    Setup:
    - 2 queries
    - Query 1 expects doc1, gets doc1 and doc2 (precision 0.5)
    - Query 2 expects doc3, gets doc3 (precision 1.0)
    Expected overall precision: 0.75
    """
    # Setup test data
    test_data = [
        TestData(
            query="test query 1",
            query_match="doc1",
            response=[
                {"id": "doc1", "vector_distance": 0.2},
                {"id": "doc2", "vector_distance": 0.3},
            ],
        ),
        TestData(
            query="test query 2",
            query_match="doc3",
            response=[
                {"id": "doc3", "vector_distance": 0.2},
                {"id": "doc4", "vector_distance": 0.8},
            ],
        ),
    ]

    # Create qrels (ground truth)
    qrels = _format_qrels(test_data)

    threshold = 0.4
    run = _generate_run_cache(test_data, threshold)

    # Calculate precision using ranx
    precision = evaluate(qrels, run, "precision")
    assert precision == 0.75  # (0.5 + 1.0) / 2


def test_known_precision_with_no_matches():
    """Test case where some queries have no matches."""
    test_data = [
        TestData(
            query="test query 2",
            query_match="",  # Expecting no match
            response=[],
        ),
    ]

    # Create qrels
    qrels = _format_qrels(test_data)

    # Generate run with threshold that excludes all docs for first query
    threshold = 0.3
    run = _generate_run_cache(test_data, threshold)

    # Calculate precision
    precision = evaluate(qrels, run, "precision")
    assert precision == 1.0  # (0.0 + 1.0) / 2
