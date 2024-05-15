import os

import pytest

from redisvl.utils.rerank import CohereReranker, HFCrossEncoderReranker


# Fixture for the reranker instance
@pytest.fixture
def cohereReranker():
    skip_reranker = os.getenv("SKIP_RERANKERS", "False").lower() == "true"
    if skip_reranker:
        pytest.skip("Skipping reranker instantiation...")
    return CohereReranker()


@pytest.fixture
def hfCrossEncoderReranker():
    return HFCrossEncoderReranker()


@pytest.fixture
def hfCrossEncoderRerankerWithCustomModel():
    return HFCrossEncoderReranker("cross-encoder/stsb-distilroberta-base")


# Test for basic ranking functionality
def test_rank_documents_cohere(cohereReranker):
    docs = ["document one", "document two", "document three"]
    query = "search query"

    reranked_docs, scores = cohereReranker.rank(query, docs)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == len(docs)  # Ensure we get back as many docs as we sent
    assert all(isinstance(score, float) for score in scores)  # Scores should be floats


# Test for asynchronous ranking functionality
@pytest.mark.asyncio
async def test_async_rank_documents_cohere(cohereReranker):
    docs = ["document one", "document two", "document three"]
    query = "search query"

    reranked_docs, scores = await cohereReranker.arank(query, docs)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == len(docs)  # Ensure we get back as many docs as we sent
    assert all(isinstance(score, float) for score in scores)  # Scores should be floats


# Test handling of bad input
def test_bad_input_cohere(cohereReranker):
    with pytest.raises(Exception):
        cohereReranker.rank("", [])  # Empty query or documents

    with pytest.raises(Exception):
        cohereReranker.rank(123, ["valid document"])  # Invalid type for query

    with pytest.raises(Exception):
        cohereReranker.rank("valid query", "not a list")  # Invalid type for documents

    with pytest.raises(Exception):
        cohereReranker.rank(
            "valid query", [{"field": "valid document"}], rank_by=["invalid_field"]
        )  # Invalid rank_by field


def test_rank_documents_cross_encoder(hfCrossEncoderReranker):
    query = "I love you"
    texts = ["I love you", "I like you", "I don't like you", "I hate you"]
    reranked_docs, scores = hfCrossEncoderReranker.rank(query, texts)

    for i in range(min(len(texts), hfCrossEncoderReranker.limit) - 1):
        assert scores[i] > scores[i + 1]


def test_rank_documents_cross_encoder_custom_model(
    hfCrossEncoderRerankerWithCustomModel,
):
    query = "I love you"
    texts = ["I love you", "I like you", "I don't like you", "I hate you"]
    reranked_docs, scores = hfCrossEncoderRerankerWithCustomModel.rank(query, texts)

    for i in range(min(len(texts), hfCrossEncoderRerankerWithCustomModel.limit) - 1):
        assert scores[i] > scores[i + 1]


@pytest.mark.asyncio
async def test_async_rank_cross_encoder(hfCrossEncoderReranker):
    docs = ["document one", "document two", "document three"]
    query = "search query"

    reranked_docs, scores = await hfCrossEncoderReranker.arank(query, docs)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == len(docs)  # Ensure we get back as many docs as we sent
    assert all(isinstance(score, float) for score in scores)  # Scores should be floats
