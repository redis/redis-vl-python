import pytest

from redisvl.utils.rerank.hf_cross_encoder import HFCrossEncoderReranker

pytestmark = pytest.mark.requires_hf


@pytest.fixture(scope="session")
def reranker():
    return HFCrossEncoderReranker()


def test_rank_documents(reranker):
    docs = ["document one", "document two", "document three"]
    query = "search query"

    reranked_docs, scores = reranker.rank(query, docs)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == reranker.limit
    assert all(isinstance(score, float) for score in scores)


@pytest.mark.asyncio
async def test_async_rank_documents(reranker):
    docs = ["document one", "document two", "document three"]
    query = "search query"

    reranked_docs, scores = await reranker.arank(query, docs)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == reranker.limit
    assert all(isinstance(score, float) for score in scores)


def test_rank_scores_align_with_reranked_docs(reranker):
    # https://github.com/redis/redis-vl-python/issues/693
    docs = [
        "irrelevant document",
        "highly relevant document",
        "somewhat relevant document",
    ]
    query = "relevant document"

    reranked_docs, scores = reranker.rank(query, docs, limit=len(docs))

    # reranked_docs is sorted by score descending, so scores must be too,
    # this catches the case where scores are taken from the original
    # unsorted list instead of the sorted one.
    assert list(scores) == sorted(scores, reverse=True)


def test_bad_input(reranker):
    with pytest.raises(ValueError):
        reranker.rank("", [])  # Empty query

    with pytest.raises(TypeError):
        reranker.rank(123, ["valid document"])  # Invalid type for query

    with pytest.raises(TypeError):
        reranker.rank("valid query", "not a list")  # Invalid type for documents


def test_rerank_empty(reranker):
    docs = []
    query = "search query"

    reranked_docs = reranker.rank(query, docs, return_score=False)

    assert isinstance(reranked_docs, list)
    assert len(reranked_docs) == 0
