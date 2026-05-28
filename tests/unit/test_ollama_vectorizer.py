import builtins
import sys
import types

import pytest

from redisvl.utils.vectorize.base import BaseVectorizer
from redisvl.utils.vectorize.text.ollama import OllamaTextVectorizer


def _embedding_for(content: str) -> list[float]:
    base = float(len(content))
    return [base, base + 1.0, base + 2.0]


@pytest.fixture
def fake_ollama_module(monkeypatch):
    fake_module = types.ModuleType("ollama")

    class FakeClient:
        instances = []
        raise_connection_error = False

        def __init__(self, host=None, **kwargs):
            self.host = host
            self.kwargs = kwargs
            self.embed_calls = []
            self.__class__.instances.append(self)

        def embed(self, model="", input="", **kwargs):
            self.embed_calls.append({"model": model, "input": input, **kwargs})
            if self.raise_connection_error:
                raise ConnectionError("daemon not running")
            if isinstance(input, str):
                return {"embeddings": [_embedding_for(input)]}
            return {"embeddings": [_embedding_for(content) for content in input]}

    class FakeAsyncClient:
        instances = []

        def __init__(self, host=None, **kwargs):
            self.host = host
            self.kwargs = kwargs
            self.embed_calls = []
            self.__class__.instances.append(self)

        async def embed(self, model="", input="", **kwargs):
            self.embed_calls.append({"model": model, "input": input, **kwargs})
            if isinstance(input, str):
                return {"embeddings": [_embedding_for(input)]}
            return {"embeddings": [_embedding_for(content) for content in input]}

    fake_module.Client = FakeClient
    fake_module.AsyncClient = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "ollama", fake_module)
    return fake_module


def test_init_sets_model_dtype_dims_type_and_clients(fake_ollama_module):
    vectorizer = OllamaTextVectorizer(
        model="nomic-embed-text",
        dtype="float64",
        host="http://localhost:11434",
        timeout=30,
    )

    sync_client = fake_ollama_module.Client.instances[0]
    async_client = fake_ollama_module.AsyncClient.instances[0]

    assert vectorizer.model == "nomic-embed-text"
    assert vectorizer.dtype == "float64"
    assert vectorizer.dims == 3
    assert vectorizer.type == "ollama"
    assert sync_client.host == "http://localhost:11434"
    assert async_client.host == "http://localhost:11434"
    assert sync_client.kwargs == {"timeout": 30}
    assert async_client.kwargs == {"timeout": 30}


def test_embed_returns_single_embedding_and_forwards_kwargs(fake_ollama_module):
    vectorizer = OllamaTextVectorizer(model="nomic-embed-text")

    result = vectorizer.embed("hello world", truncate=False, dimensions=3)

    sync_client = fake_ollama_module.Client.instances[0]
    assert result == _embedding_for("hello world")
    assert sync_client.embed_calls[-1] == {
        "model": "nomic-embed-text",
        "input": "hello world",
        "truncate": False,
        "dimensions": 3,
    }


def test_embed_many_batches_and_preserves_order(fake_ollama_module):
    vectorizer = OllamaTextVectorizer(model="nomic-embed-text")

    result = vectorizer.embed_many(["a", "bb", "ccc", "dddd"], batch_size=2)

    sync_client = fake_ollama_module.Client.instances[0]
    assert result == [
        _embedding_for("a"),
        _embedding_for("bb"),
        _embedding_for("ccc"),
        _embedding_for("dddd"),
    ]
    assert [call["input"] for call in sync_client.embed_calls[1:]] == [
        ["a", "bb"],
        ["ccc", "dddd"],
    ]


@pytest.mark.asyncio
async def test_aembed_returns_single_embedding(fake_ollama_module):
    vectorizer = OllamaTextVectorizer(model="nomic-embed-text")

    result = await vectorizer.aembed("hello async", truncate=True)

    async_client = fake_ollama_module.AsyncClient.instances[0]
    assert result == _embedding_for("hello async")
    assert async_client.embed_calls[-1] == {
        "model": "nomic-embed-text",
        "input": "hello async",
        "truncate": True,
    }


@pytest.mark.asyncio
async def test_aembed_many_returns_embeddings(fake_ollama_module):
    vectorizer = OllamaTextVectorizer(model="nomic-embed-text")

    result = await vectorizer.aembed_many(["a", "bb", "ccc"], batch_size=2)

    async_client = fake_ollama_module.AsyncClient.instances[0]
    assert result == [_embedding_for("a"), _embedding_for("bb"), _embedding_for("ccc")]
    assert [call["input"] for call in async_client.embed_calls] == [
        ["a", "bb"],
        ["ccc"],
    ]


def test_rejects_invalid_single_content(fake_ollama_module):
    vectorizer = OllamaTextVectorizer(model="nomic-embed-text")

    with pytest.raises(TypeError):
        vectorizer.embed(42)


def test_rejects_invalid_many_contents(fake_ollama_module):
    vectorizer = OllamaTextVectorizer(model="nomic-embed-text")

    with pytest.raises(TypeError):
        vectorizer.embed_many(42)

    with pytest.raises(TypeError):
        vectorizer.embed_many("not a list")

    with pytest.raises(TypeError):
        vectorizer.embed_many(["valid", 42])


@pytest.mark.asyncio
async def test_arejects_invalid_many_contents(fake_ollama_module):
    vectorizer = OllamaTextVectorizer(model="nomic-embed-text")

    with pytest.raises(TypeError):
        await vectorizer.aembed_many(42)

    with pytest.raises(TypeError):
        await vectorizer.aembed_many("not a list")

    with pytest.raises(TypeError):
        await vectorizer.aembed_many(["valid", 42])


def test_missing_ollama_dependency_raises_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ollama":
            raise ImportError("missing ollama")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "ollama", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install ollama"):
        OllamaTextVectorizer(model="nomic-embed-text")


def test_connection_error_surfaces_during_dimension_check(
    fake_ollama_module, monkeypatch
):
    fake_ollama_module.Client.raise_connection_error = True
    monkeypatch.setattr(OllamaTextVectorizer._embed.retry, "sleep", lambda _: None)

    with pytest.raises(ConnectionError, match="ollama serve|daemon"):
        OllamaTextVectorizer(model="nomic-embed-text", host="http://localhost:9999")

    assert len(fake_ollama_module.Client.instances[0].embed_calls) == 6


def test_invalid_dtype_uses_base_validation(fake_ollama_module):
    with pytest.raises(ValueError, match="Invalid data type"):
        OllamaTextVectorizer(model="nomic-embed-text", dtype="float25")


def test_uses_base_public_batch_embedding_methods():
    assert OllamaTextVectorizer.embed_many is BaseVectorizer.embed_many
    assert OllamaTextVectorizer.aembed_many is BaseVectorizer.aembed_many


def test_public_vectorize_exports_ollama_vectorizer():
    import redisvl.utils.vectorize as vectorize

    assert vectorize.OllamaTextVectorizer is OllamaTextVectorizer


def test_vectorizer_from_dict_supports_ollama(fake_ollama_module):
    from redisvl.utils.vectorize import vectorizer_from_dict

    vectorizer = vectorizer_from_dict(
        {"type": "ollama", "model": "nomic-embed-text", "dtype": "float64"}
    )

    assert isinstance(vectorizer, OllamaTextVectorizer)
    assert vectorizer.model == "nomic-embed-text"
    assert vectorizer.dtype == "float64"
