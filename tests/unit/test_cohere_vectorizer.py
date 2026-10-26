import logging
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


def test_init_creates_sync_and_async_clients(fake_cohere_module, vectorizer):
    sync_client = fake_cohere_module.Client.instances[-1]
    async_client = fake_cohere_module.AsyncClient.instances[-1]

    assert vectorizer.dims == 3
    for client in (sync_client, async_client):
        assert client.api_key == "test-key"
        assert client.client_name == "redisvl"
        assert client.kwargs == {"timeout": 30}


async def test_aembed_uses_async_client(fake_cohere_module, vectorizer, caplog):
    sync_client = fake_cohere_module.Client.instances[-1]
    async_client = fake_cohere_module.AsyncClient.instances[-1]
    sync_calls_before = len(sync_client.embed_calls)

    with caplog.at_level(logging.WARNING):
        result = await vectorizer.aembed("hello async", input_type="search_query")

    assert result == _embedding_for("hello async")
    assert async_client.embed_calls == [
        {
            "texts": ["hello async"],
            "model": "embed-english-v3.0",
            "input_type": "search_query",
            "embedding_types": ["float"],
        }
    ]
    assert len(sync_client.embed_calls) == sync_calls_before
    assert "Falling back to sync" not in caplog.text


async def test_aembed_many_batches_through_async_client(
    fake_cohere_module, vectorizer, caplog
):
    sync_client = fake_cohere_module.Client.instances[-1]
    async_client = fake_cohere_module.AsyncClient.instances[-1]
    sync_calls_before = len(sync_client.embed_calls)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    with caplog.at_level(logging.WARNING):
        result = await vectorizer.aembed_many(
            texts, batch_size=2, input_type="search_document"
        )

    assert result == [_embedding_for(text) for text in texts]
    assert [call["texts"] for call in async_client.embed_calls] == [
        ["a", "bb"],
        ["ccc", "dddd"],
        ["eeeee"],
    ]
    assert len(sync_client.embed_calls) == sync_calls_before
    assert "Falling back to sync" not in caplog.text


async def test_aembed_maps_dtype_to_embedding_types(fake_cohere_module):
    vectorizer = CohereTextVectorizer(api_config={"api_key": "test-key"}, dtype="int8")
    async_client = fake_cohere_module.AsyncClient.instances[-1]

    await vectorizer.aembed("hello", input_type="search_query")

    assert async_client.embed_calls[-1]["embedding_types"] == ["int8"]


async def test_aembed_requires_input_type(vectorizer):
    with pytest.raises(TypeError, match="input_type"):
        await vectorizer.aembed("hello")


async def test_aembed_many_rejects_non_list(vectorizer):
    with pytest.raises(TypeError, match="list of str"):
        await vectorizer.aembed_many("not a list", input_type="search_document")


async def test_aembed_ignores_embedding_types_with_warning(
    fake_cohere_module, vectorizer
):
    async_client = fake_cohere_module.AsyncClient.instances[-1]

    with pytest.warns(UserWarning, match="embedding_types"):
        await vectorizer.aembed(
            "hello", input_type="search_query", embedding_types=["int8"]
        )

    assert async_client.embed_calls[-1]["embedding_types"] == ["float"]


def test_embed_and_embed_many_still_use_sync_client(fake_cohere_module, vectorizer):
    sync_client = fake_cohere_module.Client.instances[-1]
    async_client = fake_cohere_module.AsyncClient.instances[-1]

    assert vectorizer.embed("hi", input_type="search_query") == _embedding_for("hi")
    assert vectorizer.embed_many(["x", "yy"], input_type="search_document") == [
        _embedding_for("x"),
        _embedding_for("yy"),
    ]
    assert async_client.embed_calls == []
    assert sync_client.embed_calls[-1]["texts"] == ["x", "yy"]


async def test_sync_httpx_client_keeps_async_on_sync_fallback(fake_cohere_module):
    sync_httpx = object()
    vectorizer = CohereTextVectorizer(
        api_config={"api_key": "test-key"}, httpx_client=sync_httpx
    )
    sync_client = fake_cohere_module.Client.instances[-1]

    assert sync_client.kwargs == {"httpx_client": sync_httpx}
    assert fake_cohere_module.AsyncClient.instances == []
    assert await vectorizer.aembed(
        "hello", input_type="search_query"
    ) == _embedding_for("hello")
    assert await vectorizer.aembed_many(["a", "bb"], input_type="search_document") == [
        _embedding_for("a"),
        _embedding_for("bb"),
    ]
    assert sync_client.embed_calls[-1]["texts"] == ["a", "bb"]


async def test_async_httpx_client_goes_to_async_client_only(fake_cohere_module):
    sync_httpx, async_httpx = object(), object()
    vectorizer = CohereTextVectorizer(
        api_config={"api_key": "test-key"},
        httpx_client=sync_httpx,
        async_httpx_client=async_httpx,
    )
    sync_client = fake_cohere_module.Client.instances[-1]
    async_client = fake_cohere_module.AsyncClient.instances[-1]

    assert sync_client.kwargs == {"httpx_client": sync_httpx}
    assert async_client.kwargs == {"httpx_client": async_httpx}
    await vectorizer.aembed("hello", input_type="search_query")
    assert async_client.embed_calls[-1]["texts"] == ["hello"]
