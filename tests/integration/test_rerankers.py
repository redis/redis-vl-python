import os

import pytest

from redisvl.utils.rerank import (
    CohereReranker,
    HFCrossEncoderReranker,
    VoyageAIReranker,
)


# Fixture for the reranker instance
@pytest.fixture(
    params=[
        CohereReranker,
        VoyageAIReranker,
    ]
)
def reranker(request):
    if request.param == CohereReranker:
        return request.param()
    elif request.param == VoyageAIReranker:
        return request.param(model="rerank-lite-1")


@pytest.fixture
def hfCrossEncoderReranker():
    return HFCrossEncoderReranker()


@pytest.fixture
def hfCrossEncoderRerankerWithCustomModel():
    return HFCrossEncoderReranker("cross-encoder/stsb-distilroberta-base")


@pytest.mark.requires_api_keys
def test_rank_documents(reranker):
    docs = ["document one", "document two", "document three"]
    query = "search query"

    reranked_docs, scores = reranker.rank(query, docs)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == len(docs)  # Ensure we get back as many docs as we sent
    assert all(isinstance(score, float) for score in scores)  # Scores should be floats


@pytest.mark.requires_api_keys
@pytest.mark.asyncio
async def test_async_rank_documents(reranker):
    docs = ["document one", "document two", "document three"]
    query = "search query"

    reranked_docs, scores = await reranker.arank(query, docs)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == len(docs)  # Ensure we get back as many docs as we sent
    assert all(isinstance(score, float) for score in scores)  # Scores should be floats


@pytest.mark.requires_api_keys
def test_bad_input(reranker):
    with pytest.raises(Exception):
        reranker.rank("", [])  # Empty query or documents

    with pytest.raises(Exception):
        reranker.rank(123, ["valid document"])  # Invalid type for query

    with pytest.raises(Exception):
        reranker.rank("valid query", "not a list")  # Invalid type for documents

    if isinstance(reranker, CohereReranker):
        with pytest.raises(Exception):
            reranker.rank(
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
