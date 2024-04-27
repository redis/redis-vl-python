import os

import pytest

from redisvl.utils.rerank import CohereReranker


@pytest.fixture
def skip_reranker() -> bool:
    # os.getenv returns a string
    v = os.getenv("SKIP_RERANKERS", "False").lower() == "true"
    return v


# Fixture for the reranker instance
@pytest.fixture
def reranker():
    return CohereReranker()


# Test for basic ranking functionality
def test_rank_documents(reranker, skip_reranker):
    if skip_reranker:
        pytest.skip("Skipping reranker instantiation...")
    docs = ["document one", "document two", "document three"]
    query = "search query"

    reranked_docs, scores = reranker.rank(query, docs)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == len(docs)  # Ensure we get back as many docs as we sent
    assert all(isinstance(score, float) for score in scores)  # Scores should be floats


# Test for asynchronous ranking functionality
@pytest.mark.asyncio
async def test_async_rank_documents(reranker, skip_reranker):
    if skip_reranker:
        pytest.skip("Skipping reranker instantiation...")
    docs = ["document one", "document two", "document three"]
    query = "search query"

    reranked_docs, scores = await reranker.arank(query, docs)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == len(docs)  # Ensure we get back as many docs as we sent
    assert all(isinstance(score, float) for score in scores)  # Scores should be floats


# Test handling of bad input
def test_bad_input(reranker, skip_reranker):
    if skip_reranker:
        pytest.skip("Skipping reranker instantiation...")
    with pytest.raises(Exception):
        reranker.rank("", [])  # Empty query or documents

    with pytest.raises(Exception):
        reranker.rank(123, ["valid document"])  # Invalid type for query

    with pytest.raises(Exception):
        reranker.rank("valid query", "not a list")  # Invalid type for documents

    with pytest.raises(Exception):
        reranker.rank(
            "valid query", [{"field": "valid document"}], rank_by=["invalid_field"]
        )  # Invalid rank_by field
