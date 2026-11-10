import asyncio
import sys
import types

import pytest

from redisvl.utils.rerank.cohere import CohereReranker

DOCS = ["doc one", "doc two", "doc three"]


def _response(documents, top_n):
    # Rank in reverse order with descending scores so the caller can tell the
    # stub's ordering apart from the input ordering.
    n = min(top_n or len(documents), len(documents))
    results = [
        types.SimpleNamespace(index=i, relevance_score=1.0 - i / 10)
        for i in reversed(range(len(documents)))
    ][:n]
    return types.SimpleNamespace(results=results)


@pytest.fixture
def fake_cohere_module(monkeypatch):
    fake_module = types.ModuleType("cohere")

    class FakeClient:
        instances = []

        def __init__(self, api_key=None, client_name=None, **kwargs):
            self.api_key = api_key
            self.client_name = client_name
            self.kwargs = kwargs
            self.rerank_calls = []
            self.__class__.instances.append(self)

        def rerank(self, **kwargs):
            self.rerank_calls.append(kwargs)
            return _response(kwargs["documents"], kwargs["top_n"])

    class FakeAsyncClient(FakeClient):
        instances = []

        async def rerank(self, **kwargs):
            return FakeClient.rerank(self, **kwargs)

    fake_module.Client = FakeClient
    fake_module.AsyncClient = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "cohere", fake_module)
    return fake_module


@pytest.fixture
def reranker(fake_cohere_module):
    return CohereReranker(api_config={"api_key": "test-key"}, limit=2, timeout=30)


def test_init_creates_sync_client_and_defers_async(fake_cohere_module, reranker):
    sync_client = fake_cohere_module.Client.instances[-1]

    assert sync_client.api_key == "test-key"
    assert sync_client.client_name == "redisvl"
    assert sync_client.kwargs == {"timeout": 30}
    # The async client is built on first use, inside a running event loop.
    assert fake_cohere_module.AsyncClient.instances == []


async def test_arank_uses_async_client(fake_cohere_module, reranker):
    docs, scores = await reranker.arank("query", DOCS)

    async_client = fake_cohere_module.AsyncClient.instances[-1]
    assert docs == ["doc three", "doc two"]
    assert scores == [0.8, 0.9]
    assert async_client.kwargs == {"timeout": 30}
    assert async_client.rerank_calls == [
        {
            "model": "rerank-english-v3.0",
            "query": "query",
            "top_n": 2,
            "documents": DOCS,
            "max_chunks_per_doc": None,
        }
    ]


async def test_arank_passes_rank_by_and_overrides(fake_cohere_module, reranker):
    dict_docs = [{"content": d} for d in DOCS]

    docs = await reranker.arank(
        "query", dict_docs, rank_by="content", limit=1, return_score=False
    )

    async_client = fake_cohere_module.AsyncClient.instances[-1]
    assert docs == [{"content": "doc three"}]
    call = async_client.rerank_calls[-1]
    assert call["rank_fields"] == ["content"]
    assert call["top_n"] == 1


def test_rank_still_uses_sync_client(fake_cohere_module, reranker):
    sync_client = fake_cohere_module.Client.instances[-1]

    docs, scores = reranker.rank("query", DOCS)

    assert docs == ["doc three", "doc two"]
    assert scores == [0.8, 0.9]
    assert sync_client.rerank_calls[-1]["documents"] == DOCS


def test_async_client_is_rebuilt_for_a_new_event_loop(fake_cohere_module, reranker):
    # A reranker that outlives an event loop must not reuse a client whose
    # connection pool is bound to the closed loop.
    asyncio.run(reranker.arank("first", DOCS))
    asyncio.run(reranker.arank("second", DOCS))

    first, second = fake_cohere_module.AsyncClient.instances
    assert first is not second
    assert [call["query"] for call in second.rerank_calls] == ["second"]


async def test_async_client_is_reused_within_one_event_loop(
    fake_cohere_module, reranker
):
    await reranker.arank("one", DOCS)
    await reranker.arank("two", DOCS)

    assert len(fake_cohere_module.AsyncClient.instances) == 1


async def test_sync_httpx_client_keeps_arank_on_sync_fallback(fake_cohere_module):
    sync_httpx = object()
    reranker = CohereReranker(
        api_config={"api_key": "test-key"}, limit=2, httpx_client=sync_httpx
    )
    sync_client = fake_cohere_module.Client.instances[-1]

    assert sync_client.kwargs == {"httpx_client": sync_httpx}
    docs, scores = await reranker.arank("query", DOCS)

    assert docs == ["doc three", "doc two"]
    assert scores == [0.8, 0.9]
    assert fake_cohere_module.AsyncClient.instances == []
    assert sync_client.rerank_calls[-1]["query"] == "query"


async def test_httpx_client_none_still_uses_async_client(fake_cohere_module):
    # Config dicts often carry httpx_client=None; that is not a sync client.
    reranker = CohereReranker(
        api_config={"api_key": "test-key"}, limit=2, httpx_client=None
    )

    await reranker.arank("query", DOCS)

    async_client = fake_cohere_module.AsyncClient.instances[-1]
    assert async_client.kwargs == {"httpx_client": None}
    assert async_client.rerank_calls[-1]["query"] == "query"


def test_async_httpx_client_goes_to_async_client_only_and_is_kept(
    fake_cohere_module,
):
    sync_httpx, async_httpx = object(), object()
    reranker = CohereReranker(
        api_config={"api_key": "test-key"},
        limit=2,
        httpx_client=sync_httpx,
        async_httpx_client=async_httpx,
    )
    sync_client = fake_cohere_module.Client.instances[-1]

    assert sync_client.kwargs == {"httpx_client": sync_httpx}
    asyncio.run(reranker.arank("first", DOCS))
    asyncio.run(reranker.arank("second", DOCS))

    # The caller owns this client's loop, so it is never swapped out.
    (async_client,) = fake_cohere_module.AsyncClient.instances
    assert async_client.kwargs == {"httpx_client": async_httpx}
    assert [call["query"] for call in async_client.rerank_calls] == [
        "first",
        "second",
    ]
