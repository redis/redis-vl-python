import asyncio
import sys
import types

import pytest

from redisvl.utils.vectorize.text.cohere import CohereTextVectorizer


def _embedding_for(text: str) -> list[float]:
    base = float(len(text))
    return [base, base + 1.0, base + 2.0]


def _response(texts: list[str], embedding_types: list[str]):
    vectors = [_embedding_for(text) for text in texts]
    embeddings = types.SimpleNamespace(**{t: vectors for t in embedding_types})
    return types.SimpleNamespace(embeddings=embeddings)


@pytest.fixture
def fake_cohere_module(monkeypatch):
    fake_module = types.ModuleType("cohere")

    class FakeClient:
        instances = []

        def __init__(self, api_key=None, client_name=None, **kwargs):
            self.api_key = api_key
            self.client_name = client_name
            self.kwargs = kwargs
            self.embed_calls = []
            self.__class__.instances.append(self)

        def embed(self, texts, model, input_type, embedding_types, **kwargs):
            self.embed_calls.append(
                {
                    "texts": texts,
                    "model": model,
                    "input_type": input_type,
                    "embedding_types": embedding_types,
                    **kwargs,
                }
            )
            return _response(texts, embedding_types)

    class FakeAsyncClient(FakeClient):
        instances = []

        async def embed(self, texts, model, input_type, embedding_types, **kwargs):
            return FakeClient.embed(
                self, texts, model, input_type, embedding_types, **kwargs
            )

    fake_module.Client = FakeClient
    fake_module.AsyncClient = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "cohere", fake_module)
    return fake_module


@pytest.fixture
def vectorizer(fake_cohere_module):
    return CohereTextVectorizer(api_config={"api_key": "test-key"}, timeout=30)


def test_init_creates_sync_client_and_defers_async(fake_cohere_module, vectorizer):
    sync_client = fake_cohere_module.Client.instances[-1]

    assert vectorizer.dims == 3
    assert sync_client.api_key == "test-key"
    assert sync_client.client_name == "redisvl"
    assert sync_client.kwargs == {"timeout": 30}
    # The async client is built on first use, inside a running event loop.
    assert fake_cohere_module.AsyncClient.instances == []


async def test_aembed_uses_async_client(fake_cohere_module, vectorizer):
    result = await vectorizer.aembed("hello async", input_type="search_query")

    async_client = fake_cohere_module.AsyncClient.instances[-1]
    assert result == _embedding_for("hello async")
    assert async_client.kwargs == {"timeout": 30}
    assert async_client.embed_calls == [
        {
            "texts": ["hello async"],
            "model": "embed-english-v3.0",
            "input_type": "search_query",
            "embedding_types": ["float"],
        }
    ]


async def test_aembed_many_batches_through_async_client(fake_cohere_module, vectorizer):
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    result = await vectorizer.aembed_many(
        texts, batch_size=2, input_type="search_document"
    )

    async_client = fake_cohere_module.AsyncClient.instances[-1]
    assert result == [_embedding_for(text) for text in texts]
    assert [call["texts"] for call in async_client.embed_calls] == [
        ["a", "bb"],
        ["ccc", "dddd"],
        ["eeeee"],
    ]


async def test_aembed_dtype_wins_over_embedding_types(fake_cohere_module):
    vectorizer = CohereTextVectorizer(api_config={"api_key": "test-key"}, dtype="int8")

    with pytest.warns(UserWarning, match="embedding_types"):
        await vectorizer.aembed(
            "hello", input_type="search_query", embedding_types=["float"]
        )

    async_client = fake_cohere_module.AsyncClient.instances[-1]
    assert async_client.embed_calls[-1]["embedding_types"] == ["int8"]


async def test_aembed_requires_input_type(vectorizer):
    with pytest.raises(TypeError, match="input_type"):
        await vectorizer.aembed("hello")


async def test_aembed_many_rejects_non_list(vectorizer):
    with pytest.raises(TypeError, match="list of str"):
        await vectorizer.aembed_many("not a list", input_type="search_document")


def test_embed_and_embed_many_still_use_sync_client(fake_cohere_module, vectorizer):
    sync_client = fake_cohere_module.Client.instances[-1]

    assert vectorizer.embed("hi", input_type="search_query") == _embedding_for("hi")
    assert vectorizer.embed_many(["x", "yy"], input_type="search_document") == [
        _embedding_for("x"),
        _embedding_for("yy"),
    ]
    assert sync_client.embed_calls[-1]["texts"] == ["x", "yy"]


def test_async_client_is_rebuilt_for_a_new_event_loop(fake_cohere_module, vectorizer):
    # A vectorizer that outlives an event loop must not reuse a client whose
    # connection pool is bound to the closed loop.
    asyncio.run(vectorizer.aembed("hello", input_type="search_query"))
    asyncio.run(vectorizer.aembed("again", input_type="search_query"))

    first, second = fake_cohere_module.AsyncClient.instances
    assert first is not second
    assert [call["texts"] for call in second.embed_calls] == [["again"]]


async def test_async_client_is_reused_within_one_event_loop(
    fake_cohere_module, vectorizer
):
    await vectorizer.aembed("one", input_type="search_query")
    await vectorizer.aembed_many(["two"], input_type="search_document")

    assert len(fake_cohere_module.AsyncClient.instances) == 1


async def test_sync_httpx_client_keeps_async_on_sync_fallback(fake_cohere_module):
    sync_httpx = object()
    vectorizer = CohereTextVectorizer(
        api_config={"api_key": "test-key"}, httpx_client=sync_httpx
    )
    sync_client = fake_cohere_module.Client.instances[-1]

    assert sync_client.kwargs == {"httpx_client": sync_httpx}
    assert await vectorizer.aembed(
        "hello", input_type="search_query"
    ) == _embedding_for("hello")
    assert await vectorizer.aembed_many(["a", "bb"], input_type="search_document") == [
        _embedding_for("a"),
        _embedding_for("bb"),
    ]
    assert fake_cohere_module.AsyncClient.instances == []
    assert sync_client.embed_calls[-1]["texts"] == ["a", "bb"]


async def test_httpx_client_none_still_uses_async_client(fake_cohere_module):
    # Config dicts often carry httpx_client=None; that is not a sync client.
    vectorizer = CohereTextVectorizer(
        api_config={"api_key": "test-key"}, httpx_client=None
    )

    await vectorizer.aembed("hello", input_type="search_query")

    async_client = fake_cohere_module.AsyncClient.instances[-1]
    assert async_client.kwargs == {"httpx_client": None}
    assert async_client.embed_calls[-1]["texts"] == ["hello"]


def test_async_httpx_client_goes_to_async_client_only_and_is_kept(
    fake_cohere_module,
):
    sync_httpx, async_httpx = object(), object()
    vectorizer = CohereTextVectorizer(
        api_config={"api_key": "test-key"},
        httpx_client=sync_httpx,
        async_httpx_client=async_httpx,
    )
    sync_client = fake_cohere_module.Client.instances[-1]

    assert sync_client.kwargs == {"httpx_client": sync_httpx}
    asyncio.run(vectorizer.aembed("hello", input_type="search_query"))
    asyncio.run(vectorizer.aembed("again", input_type="search_query"))

    # The caller owns this client's loop, so it is never swapped out.
    (async_client,) = fake_cohere_module.AsyncClient.instances
    assert async_client.kwargs == {"httpx_client": async_httpx}
    assert [call["texts"] for call in async_client.embed_calls] == [
        ["hello"],
        ["again"],
    ]
