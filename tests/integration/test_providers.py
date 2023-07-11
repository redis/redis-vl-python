import pytest
from redisvl.providers import (
    HuggingfaceProvider,
    OpenAIProvider
)


@pytest.fixture(params=[HuggingfaceProvider, OpenAIProvider])
def provider(request, openai_key):
    # Here we use actual models for integration test
    if request.param == HuggingfaceProvider:
        return request.param(model="sentence-transformers/all-mpnet-base-v2")
    elif request.param == OpenAIProvider:
        return request.param(model="text-embedding-ada-002", api_config={
            "api_key": openai_key
            })

def test_provider_embed(provider):
    text = 'This is a test sentence.'
    embedding = provider.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == provider.dims

def test_provider_embed_many(provider):
    texts = ['This is the first test sentence.', 'This is the second test sentence.']
    embeddings = provider.embed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) and len(emb) == provider.dims for emb in embeddings)


@pytest.fixture(params=[OpenAIProvider])
def aprovider(request, openai_key):
    # Here we use actual models for integration test
    if request.param == OpenAIProvider:
        return request.param(model="text-embedding-ada-002", api_config={
            "api_key": openai_key
            })

@pytest.mark.asyncio
async def test_provider_aembed(aprovider):
    text = 'This is a test sentence.'
    embedding = await aprovider.aembed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == aprovider.dims

@pytest.mark.asyncio
async def test_provider_aembed_many(aprovider):
    texts = ['This is the first test sentence.', 'This is the second test sentence.']
    embeddings = await aprovider.aembed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) and len(emb) == aprovider.dims for emb in embeddings)
