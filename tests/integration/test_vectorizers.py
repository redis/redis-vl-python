import os

import pytest

from redisvl.vectorize.text import (
    HFTextVectorizer,
    OpenAITextVectorizer,
    VertexAITextVectorizer,
    CohereTextVectorizer
)


@pytest.fixture
def skip_vectorizer() -> bool:
    # os.getenv returns a string
    v = os.getenv("SKIP_VECTORIZERS", "False").lower() == "true"
    print(v, flush=True)
    return v


skip_vectorizer_test = lambda: pytest.config.getfixturevalue("skip_vectorizer")


@pytest.fixture(params=[HFTextVectorizer, OpenAITextVectorizer, VertexAITextVectorizer, CohereTextVectorizer])
def vectorizer(request):
    if request.param == HFTextVectorizer:
        return request.param()
    elif request.param == OpenAITextVectorizer:
        return request.param()
    elif request.param == VertexAITextVectorizer:
        return request.param()
    elif request.param == CohereTextVectorizer:
        return request.param(
            model="embed-english-v3.0", api_config={"api_key": cohere_key}
        )

# @pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
@pytest.mark.run
def test_vectorizer_embed(vectorizer):
    text = "This is a test sentence."
    if isinstance(vectorizer, CohereTextVectorizer):
        embedding = vectorizer.embed(text, input_type="search_document")
    else:
        embedding = vectorizer.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims


@pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
def test_vectorizer_embed_many(vectorizer):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    if isinstance(vectorizer, CohereTextVectorizer):
        embeddings = vectorizer.embed_many(texts, input_type="search_document")
    else:
        embeddings = vectorizer.embed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == vectorizer.dims for emb in embeddings
    )


@pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
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
        return request.param()

# @pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
@pytest.mark.asyncio
async def test_vectorizer_aembed(avectorizer):
    text = "This is a test sentence."
    embedding = await avectorizer.aembed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == avectorizer.dims


# @pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
@pytest.mark.asyncio
async def test_vectorizer_aembed_many(avectorizer):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    embeddings = await avectorizer.aembed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == avectorizer.dims for emb in embeddings
    )


# @pytest.mark.skipif(skip_vectorizer_test, reason="Skipping vectorizer tests")
@pytest.mark.asyncio
async def test_avectorizer_bad_input(avectorizer):
    with pytest.raises(TypeError):
        avectorizer.embed(1)

    with pytest.raises(TypeError):
        avectorizer.embed({"foo": "bar"})

    with pytest.raises(TypeError):
        avectorizer.embed_many(42)
