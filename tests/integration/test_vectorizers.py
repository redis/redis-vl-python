import os

import pytest

from redisvl.utils.vectorize import (
    AzureOpenAITextVectorizer,
    CohereTextVectorizer,
    HFTextVectorizer,
    OpenAITextVectorizer,
    VertexAITextVectorizer,
    VoyageAITextVectorizer,
)


@pytest.fixture
def skip_vectorizer() -> bool:
    # os.getenv returns a string
    v = os.getenv("SKIP_VECTORIZERS", "False").lower() == "true"
    return v


@pytest.fixture(
    params=[
        HFTextVectorizer,
        OpenAITextVectorizer,
        VertexAITextVectorizer,
        CohereTextVectorizer,
        AzureOpenAITextVectorizer,
        VoyageAITextVectorizer,
    ]
)
def vectorizer(request, skip_vectorizer):
    if skip_vectorizer:
        pytest.skip("Skipping vectorizer instantiation...")

    if request.param == HFTextVectorizer:
        return request.param()
    elif request.param == OpenAITextVectorizer:
        return request.param()
    elif request.param == VertexAITextVectorizer:
        return request.param()
    elif request.param == CohereTextVectorizer:
        return request.param()
    elif request.param == VoyageAITextVectorizer:
        return request.param(model="voyage-large-2")
    elif request.param == AzureOpenAITextVectorizer:
        return request.param(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-ada-002")
        )


def test_vectorizer_embed(vectorizer):
    text = "This is a test sentence."
    if isinstance(vectorizer, CohereTextVectorizer):
        embedding = vectorizer.embed(text, input_type="search_document")
    elif isinstance(vectorizer, VoyageAITextVectorizer):
        embedding = vectorizer.embed(text, input_type="document")
    else:
        embedding = vectorizer.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims


def test_vectorizer_embed_many(vectorizer):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    if isinstance(vectorizer, CohereTextVectorizer):
        embeddings = vectorizer.embed_many(texts, input_type="search_document")
    elif isinstance(vectorizer, VoyageAITextVectorizer):
        embeddings = vectorizer.embed_many(texts, input_type="document")
    else:
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


@pytest.fixture(params=[OpenAITextVectorizer, VoyageAITextVectorizer])
def avectorizer(request, skip_vectorizer):
    if skip_vectorizer:
        pytest.skip("Skipping vectorizer instantiation...")

    # Here we use actual models for integration test
    if request.param == OpenAITextVectorizer:
        return request.param()
    elif request.param == VoyageAITextVectorizer:
        return request.param(model="voyage-large-2")


@pytest.mark.asyncio
async def test_vectorizer_aembed(avectorizer):
    text = "This is a test sentence."
    if isinstance(avectorizer, VoyageAITextVectorizer):
        embedding = await avectorizer.aembed(text)
    else:
        embedding = await avectorizer.aembed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == avectorizer.dims


@pytest.mark.asyncio
async def test_vectorizer_aembed_many(avectorizer):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    if isinstance(avectorizer, VoyageAITextVectorizer):
        embeddings = await avectorizer.aembed_many(texts)
    else:
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
