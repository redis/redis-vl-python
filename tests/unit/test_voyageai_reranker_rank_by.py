import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def reranker(monkeypatch):
    fake = types.ModuleType("voyageai")
    fake.Client = MagicMock()
    fake.AsyncClient = MagicMock()
    monkeypatch.setitem(sys.modules, "voyageai", fake)
    from redisvl.utils.rerank.voyageai import VoyageAIReranker

    return VoyageAIReranker(
        model="rerank-2", api_config={"api_key": "k"}, rank_by=["text"]
    )


DOCS = [{"text": "alpha", "id": 1}, {"text": "beta", "id": 2}]


def test_rank_by_selects_field(reranker):
    kwargs, _ = reranker._preprocess("q", DOCS)
    assert kwargs["documents"] == ["alpha", "beta"]


def test_rank_by_multiple_fields_and_override(reranker):
    docs = [{"text": "alpha", "title": "T1"}, {"text": "beta", "title": "T2"}]
    kwargs, _ = reranker._preprocess("q", docs, rank_by=["title", "text"])
    assert kwargs["documents"] == ["T1 alpha", "T2 beta"]


def test_rank_by_missing_field_raises(reranker):
    with pytest.raises(ValueError, match="rank_by"):
        reranker._preprocess("q", [{"other": "x"}])


def test_content_key_still_default(monkeypatch):
    fake = types.ModuleType("voyageai")
    fake.Client = MagicMock()
    fake.AsyncClient = MagicMock()
    monkeypatch.setitem(sys.modules, "voyageai", fake)
    from redisvl.utils.rerank.voyageai import VoyageAIReranker

    r = VoyageAIReranker(model="rerank-2", api_config={"api_key": "k"})
    kwargs, _ = r._preprocess("q", [{"content": "a"}, {"content": "b"}])
    assert kwargs["documents"] == ["a", "b"]
