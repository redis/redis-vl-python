import os

import pytest

from redisvl.vectorize.text import (
    HFTextVectorizer,
    OpenAITextVectorizer,
    VertexAITextVectorizer,
)


@pytest.fixture(params=[HFTextVectorizer, OpenAITextVectorizer, VertexAITextVectorizer])
def vectorizer(request, openai_key):
    # Here we use actual models for integration test
    if request.param == HFTextVectorizer:
        return request.param(model="sentence-transformers/all-mpnet-base-v2")
    elif request.param == OpenAITextVectorizer:
        return request.param(
            model="text-embedding-ada-002", api_config={"api_key": openai_key}
        )
    elif request.param == VertexAITextVectorizer:
        # also need to set GOOGLE_APPLICATION_CREDENTIALS env var
        return request.param(
            model="textembedding-gecko",
            api_config={
                "location": os.environ["LOCATION"],
                "project": os.environ["PROJECT"],
            },
        )


def test_vectorizer_embed(vectorizer):
    text = "This is a test sentence."
    embedding = vectorizer.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims


def test_vectorizer_embed_many(vectorizer):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    embeddings = vectorizer.embed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == vectorizer.dims for emb in embeddings
    )


def test_vectorizer_bad_input(vectorizer):
    with pytest.raises(TypeError):
        vectorizer.embed(1)

    with pytest.raises(TypeError):
        vectorizer.embed({"foo": "bar"})

    with pytest.raises(TypeError):
        vectorizer.embed_many(42)


@pytest.fixture(params=[OpenAITextVectorizer])
def avectorizer(request, openai_key):
    # Here we use actual models for integration test
    if request.param == OpenAITextVectorizer:
        return request.param(
            model="text-embedding-ada-002", api_config={"api_key": openai_key}
        )


@pytest.mark.asyncio
async def test_vectorizer_aembed(avectorizer):
    text = "This is a test sentence."
    embedding = await avectorizer.aembed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == avectorizer.dims


@pytest.mark.asyncio
async def test_vectorizer_aembed_many(avectorizer):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    embeddings = await avectorizer.aembed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == avectorizer.dims for emb in embeddings
    )


@pytest.mark.asyncio
async def test_avectorizer_bad_input(avectorizer):
    with pytest.raises(TypeError):
        avectorizer.embed(1)

    with pytest.raises(TypeError):
        avectorizer.embed({"foo": "bar"})

    with pytest.raises(TypeError):
        avectorizer.embed_many(42)
