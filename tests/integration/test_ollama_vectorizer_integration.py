import os

import pytest

from redisvl.utils.vectorize.text.ollama import OllamaTextVectorizer

pytestmark = pytest.mark.skipif(
    os.getenv("REDISVL_TEST_OLLAMA") != "1",
    reason=("Set REDISVL_TEST_OLLAMA=1 with Ollama running and the model pulled."),
)


@pytest.fixture(scope="module")
def vectorizer():
    return OllamaTextVectorizer(
        model=os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
        host=os.getenv("OLLAMA_HOST"),
    )


def _assert_valid_embedding(embedding, dims):
    assert isinstance(embedding, list)
    assert len(embedding) == dims
    assert all(isinstance(value, (float, int)) for value in embedding)


def test_real_embed_returns_vector(vectorizer):
    embedding = vectorizer.embed("hello world")

    _assert_valid_embedding(embedding, vectorizer.dims)


def test_real_embed_many_returns_one_vector_per_input(vectorizer):
    embeddings = vectorizer.embed_many(["a", "b", "c"])

    assert len(embeddings) == 3
    for embedding in embeddings:
        _assert_valid_embedding(embedding, vectorizer.dims)


@pytest.mark.asyncio
async def test_real_async_embed(vectorizer):
    embedding = await vectorizer.aembed("hello world")

    _assert_valid_embedding(embedding, vectorizer.dims)


@pytest.mark.asyncio
async def test_real_async_embed_many_returns_one_vector_per_input(vectorizer):
    embeddings = await vectorizer.aembed_many(["a", "b", "c"])

    assert len(embeddings) == 3
    for embedding in embeddings:
        _assert_valid_embedding(embedding, vectorizer.dims)
