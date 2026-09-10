import os
import warnings

import numpy as np
import pytest

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
from redisvl.utils.utils import create_ulid
from redisvl.utils.vectorize import (
    AzureOpenAITextVectorizer,
    BedrockVectorizer,
    CohereTextVectorizer,
    CustomVectorizer,
    GoogleGenAIVectorizer,
    HFTextVectorizer,
    MistralAITextVectorizer,
    OpenAITextVectorizer,
    VertexAIVectorizer,
    VoyageAIVectorizer,
)

# Constants for testing
TEST_TEXT = "This is a test sentence."
TEST_TEXTS = ["This is the first test sentence.", "This is the second test sentence."]
TEST_VECTOR = [1.1, 2.2, 3.3, 4.4]


@pytest.fixture
def embeddings_cache(client):
    """Create a real EmbeddingsCache for testing with a unique namespace."""
    # Use a unique prefix for this test run to avoid conflicts
    unique_prefix = f"test_cache_{create_ulid()}"

    # Create the cache with a short TTL
    cache = EmbeddingsCache(name=unique_prefix, ttl=10, redis_client=client)

    yield cache

    cache.clear()


# Azure OpenAI live tests need a reachable deployment. _initialize_clients()
# requires all three of these before it will construct a client, so gate on all
# three rather than guessing from one. AZURE_OPENAI_DEPLOYMENT_NAME is
# deliberately excluded -- it has a real default at every call site below.
_AZURE_CONFIGURED = all(
    os.getenv(var)
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "OPENAI_API_VERSION")
)
skip_without_azure = pytest.mark.skipif(
    not _AZURE_CONFIGURED,
    reason=(
        "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT, "
        "AZURE_OPENAI_API_KEY and OPENAI_API_VERSION to run these, plus "
        "AZURE_OPENAI_DEPLOYMENT_NAME if your deployment is not named "
        "text-embedding-ada-002. Offline coverage lives in "
        "tests/unit/test_azure_openai_vectorizer.py."
    ),
)


_vectorizer_params = [
    pytest.param(HFTextVectorizer, marks=pytest.mark.requires_hf),
    OpenAITextVectorizer,
    VertexAIVectorizer,
    GoogleGenAIVectorizer,
    CohereTextVectorizer,
    pytest.param(AzureOpenAITextVectorizer, marks=skip_without_azure),
    BedrockVectorizer,
    MistralAITextVectorizer,
    CustomVectorizer,
    VoyageAIVectorizer,
]


@pytest.fixture(params=_vectorizer_params)
def vectorizer(request):
    if request.param == HFTextVectorizer:
        return request.param()
    elif request.param == OpenAITextVectorizer:
        return request.param()
    elif request.param == VertexAIVectorizer:
        return request.param()
    elif request.param == GoogleGenAIVectorizer:
        # Prefer the Gemini Developer API key when present so this fixture
        # exercises the Gemini backend. The dtype-param tests below construct
        # GoogleGenAIVectorizer() with no config, exercising env auto-detect
        # (Vertex when GCP creds are set) — so both backends get covered in CI.
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            return request.param(api_config={"api_key": api_key})
        return request.param()
    elif request.param == CohereTextVectorizer:
        return request.param()
    elif request.param == MistralAITextVectorizer:
        return request.param()
    elif request.param == VoyageAIVectorizer:
        return request.param(model="voyage-large-2")
    elif request.param == AzureOpenAITextVectorizer:
        return request.param(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-ada-002")
        )
    elif request.param == BedrockVectorizer:
        return request.param(
            model=os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0")
        )
    elif request.param == CustomVectorizer:

        def embed(content):
            return TEST_VECTOR

        def embed_many(contents):
            return [TEST_VECTOR] * len(contents)

        async def aembed_func(content):
            return TEST_VECTOR

        async def aembed_many_func(contents):
            return [TEST_VECTOR] * len(contents)

        return request.param(embed=embed, embed_many=embed_many)


@pytest.fixture
def cached_vectorizer(embeddings_cache):
    """Create a simple custom vectorizer for testing."""

    def embed(content):
        return TEST_VECTOR

    def embed_many(contents):
        return [TEST_VECTOR] * len(contents)

    async def aembed(content):
        return TEST_VECTOR

    async def aembed_many(contents):
        return [TEST_VECTOR] * len(contents)

    return CustomVectorizer(
        embed=embed,
        embed_many=embed_many,
        aembed=aembed,
        aembed_many=aembed_many,
        cache=embeddings_cache,
    )


@pytest.fixture
def custom_embed_func():
    def embed(content: str):
        return TEST_VECTOR

    return embed


@pytest.fixture
def custom_embed_class():
    class MyEmbedder:
        def embed(self, content: str):
            return TEST_VECTOR

        def embed_with_args(self, content: str, max_len=None):
            return TEST_VECTOR[0:max_len]

        def embed_many(self, contents):
            return [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

        def embed_many_with_args(self, contents, param=True):
            if param:
                return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            else:
                return [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]

    return MyEmbedder


@pytest.mark.requires_api_keys
def test_vectorizer_embed(vectorizer):
    text = TEST_TEXT
    if isinstance(vectorizer, CohereTextVectorizer):
        embedding = vectorizer.embed(text, input_type="search_document")
    elif isinstance(vectorizer, VoyageAIVectorizer):
        embedding = vectorizer.embed(text, input_type="document")
    else:
        embedding = vectorizer.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims


@pytest.mark.requires_api_keys
def test_vectorizer_embed_many(vectorizer):
    texts = TEST_TEXTS
    if isinstance(vectorizer, CohereTextVectorizer):
        embeddings = vectorizer.embed_many(texts, input_type="search_document")
    elif isinstance(vectorizer, VoyageAIVectorizer):
        embeddings = vectorizer.embed_many(texts, input_type="document")
    else:
        embeddings = vectorizer.embed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == vectorizer.dims for emb in embeddings
    )


def test_vectorizer_with_cache(cached_vectorizer):
    """Test the complete cache flow - miss, store, hit."""
    # First call - should be a cache miss
    first_result = cached_vectorizer.embed(TEST_TEXT)
    assert first_result == TEST_VECTOR

    # Second call - should be a cache hit
    second_result = cached_vectorizer.embed(TEST_TEXT)
    assert second_result == TEST_VECTOR

    # Verify it's actually using the cache by checking the cached value exists
    cached_entry = cached_vectorizer.cache.get(
        content=TEST_TEXT, model_name=cached_vectorizer.model
    )
    assert cached_entry is not None
    assert cached_entry["embedding"] == TEST_VECTOR


def test_vectorizer_with_cache_skip(cached_vectorizer):
    """Test embedding with skip_cache=True."""
    # Store a value in the cache
    cached_vectorizer.embed(TEST_TEXT)

    # Call embed with skip_cache=True - should bypass cache
    cached_vectorizer.cache.drop(content=TEST_TEXT, model_name=cached_vectorizer.model)

    # Store a deliberately different value in the cache
    cached_vectorizer.cache.set(
        content=TEST_TEXT,
        model_name=cached_vectorizer.model,
        embedding=[9.9, 8.8, 7.7, 6.6],
    )

    # Now call with skip_cache=True
    result = cached_vectorizer.embed(TEST_TEXT, skip_cache=True)

    # Should generate fresh result, not use cached value
    assert result == TEST_VECTOR

    # Cache should still have the original value
    cached_entry = cached_vectorizer.cache.get(
        content=TEST_TEXT, model_name=cached_vectorizer.model
    )
    assert cached_entry["embedding"] == [9.9, 8.8, 7.7, 6.6]


def test_vectorizer_with_cache_many(cached_vectorizer):
    """Test embedding many texts with partial cache hits/misses."""
    # Store an embedding for the first text only
    cached_vectorizer.cache.set(
        content=TEST_TEXTS[0],
        model_name=cached_vectorizer.model,
        embedding=[0.1, 0.2, 0.3, 0.4],
    )

    # Call embed_many - should hit cache for first text, miss for second
    results = cached_vectorizer.embed_many(TEST_TEXTS)

    # Verify results
    assert results[0] == [0.1, 0.2, 0.3, 0.4]  # From cache
    assert results[1] == TEST_VECTOR  # Generated

    # Both should now be in cache
    for text in TEST_TEXTS:
        assert cached_vectorizer.cache.exists(
            content=text, model_name=cached_vectorizer.model
        )


def test_vectorizer_with_cached_metadata(cached_vectorizer):
    """Test passing metadata through to the cache."""
    # Call embed with metadata
    test_metadata = {"source": "test", "importance": "high"}
    cached_vectorizer.embed(TEST_TEXT, metadata=test_metadata)

    # Verify metadata was stored in cache
    cached_entry = cached_vectorizer.cache.get(
        content=TEST_TEXT, model_name=cached_vectorizer.model
    )
    assert cached_entry["metadata"] == test_metadata


@pytest.mark.asyncio
async def test_vectorizer_with_cache_async(cached_vectorizer):
    """Test async embedding with cache."""
    # First call - should be a cache miss
    first_result = await cached_vectorizer.aembed(TEST_TEXT)
    assert first_result == TEST_VECTOR

    # Second call - should be a cache hit
    second_result = await cached_vectorizer.aembed(TEST_TEXT)
    assert second_result == TEST_VECTOR

    # Verify it's actually using the cache
    cached_entry = await cached_vectorizer.cache.aget(
        content=TEST_TEXT, model_name=cached_vectorizer.model
    )
    assert cached_entry is not None
    assert cached_entry["embedding"] == TEST_VECTOR


@pytest.mark.asyncio
async def test_vectorizer_with_cache_async_many(cached_vectorizer):
    """Test async embedding many texts with partial cache hits/misses."""
    # Store an embedding for the first text only
    await cached_vectorizer.cache.aset(
        content=TEST_TEXTS[0],
        model_name=cached_vectorizer.model,
        embedding=[0.1, 0.2, 0.3, 0.4],
    )

    # Call aembed_many - should hit cache for first text, miss for second
    results = await cached_vectorizer.aembed_many(TEST_TEXTS)

    # Verify results
    assert results[0] == [0.1, 0.2, 0.3, 0.4]  # From cache
    assert results[1] == TEST_VECTOR  # Generated

    # Both should now be in cache
    for text in TEST_TEXTS:
        assert await cached_vectorizer.cache.aexists(
            content=text, model_name=cached_vectorizer.model
        )


@pytest.mark.requires_api_keys
def test_bedrock_bad_credentials():
    with pytest.raises(ValueError):
        BedrockVectorizer(
            api_config={
                "aws_access_key_id": "invalid",
                "aws_secret_access_key": "invalid",
            }
        )


@pytest.mark.requires_api_keys
def test_bedrock_invalid_model():
    with pytest.raises(ValueError):
        bedrock = BedrockVectorizer(model="invalid-model")
        bedrock.embed("test")


def test_custom_vectorizer_embed(custom_embed_class, custom_embed_func):
    custom_wrapper = CustomVectorizer(embed=custom_embed_func)
    embedding = custom_wrapper.embed("This is a test sentence.")
    assert embedding == TEST_VECTOR

    custom_wrapper = CustomVectorizer(embed=custom_embed_class().embed)
    embedding = custom_wrapper.embed("This is a test sentence.")
    assert embedding == TEST_VECTOR

    custom_wrapper = CustomVectorizer(embed=custom_embed_class().embed_with_args)
    embedding = custom_wrapper.embed("This is a test sentence.", max_len=4)
    assert embedding == TEST_VECTOR
    embedding = custom_wrapper.embed("This is a test sentence.", max_len=2)
    assert embedding == [1.1, 2.2]

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(embed="hello")

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(embed=42)

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(embed={"foo": "bar"})

    def bad_arg_type(value: int):
        return [value]

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(embed=bad_arg_type)

    def bad_return_type(text: str) -> str:
        return text

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(embed=bad_return_type)


def test_custom_vectorizer_embed_many(custom_embed_class, custom_embed_func):
    custom_wrapper = CustomVectorizer(
        custom_embed_func, embed_many=custom_embed_class().embed_many
    )
    embeddings = custom_wrapper.embed_many(["test one.", "test two"])
    assert embeddings == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    custom_wrapper = CustomVectorizer(
        custom_embed_func, embed_many=custom_embed_class().embed_many
    )
    embeddings = custom_wrapper.embed_many(["test one.", "test two"])
    assert embeddings == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    custom_wrapper = CustomVectorizer(
        custom_embed_func, embed_many=custom_embed_class().embed_many_with_args
    )
    embeddings = custom_wrapper.embed_many(["test one.", "test two"], param=True)
    assert embeddings == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    embeddings = custom_wrapper.embed_many(["test one.", "test two"], param=False)
    assert embeddings == [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(custom_embed_func, embed_many="hello")

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(custom_embed_func, embed_many=42)

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(
            custom_embed_func, embed_many={"foo": "bar"}
        )

    def bad_arg_type(value: int):
        return [value]

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(
            custom_embed_func, embed_many=bad_arg_type
        )

    def bad_return_type(text: str) -> str:
        return text

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomVectorizer(
            custom_embed_func, embed_many=bad_return_type
        )


_dtype_params = [
    pytest.param(AzureOpenAITextVectorizer, marks=skip_without_azure),
    BedrockVectorizer,
    CohereTextVectorizer,
    CustomVectorizer,
    pytest.param(HFTextVectorizer, marks=pytest.mark.requires_hf),
    MistralAITextVectorizer,
    OpenAITextVectorizer,
    VertexAIVectorizer,
    GoogleGenAIVectorizer,
    VoyageAIVectorizer,
]


@pytest.mark.requires_api_keys
@pytest.mark.parametrize("vectorizer_", _dtype_params)
def test_default_dtype(vectorizer_):
    # test dtype defaults to float32
    if issubclass(vectorizer_, CustomVectorizer):
        vectorizer = vectorizer_(embed=lambda x, input_type=None: [1.0, 2.0, 3.0])
    elif issubclass(vectorizer_, AzureOpenAITextVectorizer):
        vectorizer = vectorizer_(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-ada-002")
        )
    else:
        vectorizer = vectorizer_()

    assert vectorizer.dtype == "float32"


@pytest.mark.requires_api_keys
@pytest.mark.parametrize("vectorizer_", _dtype_params)
def test_vectorizer_dtype_assignment(vectorizer_):
    # test initializing dtype in constructor
    for dtype in ["float16", "float32", "float64", "bfloat16", "int8", "uint8"]:
        if issubclass(vectorizer_, CustomVectorizer):
            vectorizer = vectorizer_(embed=lambda x: [1.0, 2.0, 3.0], dtype=dtype)
        elif issubclass(vectorizer_, AzureOpenAITextVectorizer):
            vectorizer = vectorizer_(
                model=os.getenv(
                    "AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-ada-002"
                ),
                dtype=dtype,
            )
        else:
            vectorizer = vectorizer_(dtype=dtype)

        assert vectorizer.dtype == dtype


# Params for non-supported dtype tests (no CustomVectorizer here)
_non_supported_dtype_params = [
    AzureOpenAITextVectorizer,
    BedrockVectorizer,
    CohereTextVectorizer,
    pytest.param(HFTextVectorizer, marks=pytest.mark.requires_hf),
    MistralAITextVectorizer,
    OpenAITextVectorizer,
    VertexAIVectorizer,
    GoogleGenAIVectorizer,
    VoyageAIVectorizer,
]


@pytest.mark.requires_api_keys
@pytest.mark.parametrize("vectorizer_", _non_supported_dtype_params)
def test_non_supported_dtypes(vectorizer_):
    with pytest.raises(ValueError):
        vectorizer_(dtype="float25")

    with pytest.raises(ValueError):
        vectorizer_(dtype=7)

    with pytest.raises(ValueError):
        vectorizer_(dtype=None)


@pytest.mark.requires_api_keys
@pytest.mark.asyncio
async def test_vectorizer_aembed(vectorizer):
    text = TEST_TEXT
    if isinstance(vectorizer, CohereTextVectorizer):
        embedding = await vectorizer.aembed(text, input_type="search_document")
    elif isinstance(vectorizer, VoyageAIVectorizer):
        embedding = await vectorizer.aembed(text, input_type="document")
    else:
        embedding = await vectorizer.aembed(text)
    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims


@pytest.mark.requires_api_keys
@pytest.mark.asyncio
async def test_vectorizer_aembed_many(vectorizer):
    texts = TEST_TEXTS
    if isinstance(vectorizer, CohereTextVectorizer):
        embeddings = await vectorizer.aembed_many(texts, input_type="search_document")
    elif isinstance(vectorizer, VoyageAIVectorizer):
        embeddings = await vectorizer.aembed_many(texts, input_type="document")
    else:
        embeddings = await vectorizer.aembed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == vectorizer.dims for emb in embeddings
    )


@pytest.mark.requires_api_keys
@pytest.mark.parametrize(
    "dtype,expected_type",
    [
        ("float32", float),  # Float dtype should return floats
        ("int8", int),  # Int8 dtype should return ints
        ("uint8", int),  # Uint8 dtype should return ints
    ],
)
def test_cohere_dtype_support(dtype, expected_type):
    """Test that CohereTextVectorizer properly handles different dtypes for embeddings."""
    text = TEST_TEXT
    texts = TEST_TEXTS

    # Create vectorizer with specified dtype
    vectorizer = CohereTextVectorizer(dtype=dtype)

    # Verify the correct mapping of dtype to Cohere embedding_types
    if dtype == "int8":
        assert vectorizer._get_cohere_embedding_type(dtype) == ["int8"]
    elif dtype == "uint8":
        assert vectorizer._get_cohere_embedding_type(dtype) == ["uint8"]
    else:
        # All other dtypes should map to float
        assert vectorizer._get_cohere_embedding_type(dtype) == ["float"]

    # Test single embedding
    embedding = vectorizer.embed(text, input_type="search_document")
    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims

    # Check that all elements are of the expected type
    assert all(
        isinstance(val, expected_type) for val in embedding
    ), f"Expected all elements to be {expected_type.__name__} for dtype {dtype}"

    # Test multiple embeddings
    embeddings = vectorizer.embed_many(texts, input_type="search_document")
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == vectorizer.dims for emb in embeddings
    )

    # Check that all elements in all embeddings are of the expected type
    for emb in embeddings:
        assert all(
            isinstance(val, expected_type) for val in emb
        ), f"Expected all elements to be {expected_type.__name__} for dtype {dtype}"

    # Test as_buffer output format
    embedding_buffer = vectorizer.embed(
        text, input_type="search_document", as_buffer=True
    )
    assert isinstance(embedding_buffer, bytes)

    # Test embed_many with as_buffer=True
    buffer_embeddings = vectorizer.embed_many(
        texts, input_type="search_document", as_buffer=True
    )
    assert all(isinstance(emb, bytes) for emb in buffer_embeddings)

    # Compare dimensions between buffer and list formats
    assert len(np.frombuffer(embedding_buffer, dtype=dtype)) == len(embedding)


@pytest.mark.requires_api_keys
def test_cohere_embedding_types_warning():
    """Test that a warning is raised when embedding_types parameter is passed."""
    text = TEST_TEXT
    texts = TEST_TEXTS
    vectorizer = CohereTextVectorizer()

    # Test warning for single embedding
    with pytest.warns(UserWarning, match="embedding_types.*not supported"):
        embedding = vectorizer.embed(
            text,
            input_type="search_document",
            embedding_types=["uint8"],  # explicitly testing the anti-pattern here
        )
    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims

    # Test warning for multiple embeddings
    with pytest.warns(UserWarning, match="embedding_types.*not supported"):
        embeddings = vectorizer.embed_many(
            texts, input_type="search_document", embedding_types=["uint8"]
        )
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)


@pytest.mark.requires_hf
def test_deprecated_text_parameter_warning():
    """Test that using deprecated 'text' and 'texts' parameters emits deprecation warnings."""
    vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

    # Test single embed with deprecated 'text' parameter emits warning
    with pytest.warns(DeprecationWarning, match="Argument text is deprecated"):
        embedding = vectorizer.embed(text=TEST_TEXT)
    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims

    # Test embed_many with deprecated 'texts' parameter emits warning
    with pytest.warns(DeprecationWarning, match="Argument texts is deprecated"):
        embeddings = vectorizer.embed_many(texts=TEST_TEXTS)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(TEST_TEXTS)


# --- VoyageAI model-routing tests (mocked, no API key required) ---


def _fake_voyage_clients(dims=4):
    """Build mocked sync/async VoyageAI clients returning fixed-size embeddings.

    Each endpoint returns exactly one embedding per requested input so tests can
    assert that batching never collapses multiple inputs into a single request.
    """
    from unittest.mock import AsyncMock, MagicMock

    def _ctx_embed(inputs, model, input_type=None, **kwargs):
        resp = MagicMock()
        # contextualized_embed returns one result per input document, each with
        # a list of per-chunk embeddings.
        resp.results = [
            MagicMock(index=i, embeddings=[[0.1] * dims]) for i in range(len(inputs))
        ]
        return resp

    def _embed(texts, model=None, input_type=None, truncation=True, **kwargs):
        resp = MagicMock()
        resp.embeddings = [[0.1] * dims for _ in texts]
        return resp

    def _mm_embed(inputs, model, input_type=None, truncation=True, **kwargs):
        resp = MagicMock()
        resp.embeddings = [[0.1] * dims for _ in inputs]
        return resp

    def _tokenize(texts, model=None):
        # One token per whitespace-delimited word, so tests can control token
        # counts by word count (mirrors voyageai's tokenize return shape).
        return [text.split() for text in texts]

    client = MagicMock()
    client.contextualized_embed.side_effect = _ctx_embed
    client.embed.side_effect = _embed
    client.multimodal_embed.side_effect = _mm_embed
    client.tokenize.side_effect = _tokenize

    aclient = MagicMock()
    aclient.contextualized_embed = AsyncMock(side_effect=_ctx_embed)
    aclient.embed = AsyncMock(side_effect=_embed)
    aclient.multimodal_embed = AsyncMock(side_effect=_mm_embed)
    aclient.tokenize.side_effect = _tokenize

    return client, aclient


def _build_voyage_vectorizer(model):
    from unittest.mock import patch

    client, aclient = _fake_voyage_clients()
    with (
        patch("voyageai.Client", return_value=client),
        patch("voyageai.AsyncClient", return_value=aclient),
    ):
        vectorizer = VoyageAIVectorizer(model=model, api_config={"api_key": "test"})
    return vectorizer, client, aclient


def test_voyageai_context_model_detection():
    """voyage-context-* models are detected as contextualized models."""
    ctx_vectorizer, _, _ = _build_voyage_vectorizer("voyage-context-4")
    assert ctx_vectorizer.is_context is True
    assert ctx_vectorizer.is_multimodal is False

    plain_vectorizer, _, _ = _build_voyage_vectorizer("voyage-3-large")
    assert plain_vectorizer.is_context is False


def test_voyageai_context_embed_many_uses_contextualized_api():
    """Context models route to contextualized_embed with auto-chunking enabled."""
    vectorizer, client, _ = _build_voyage_vectorizer("voyage-context-4")

    embeddings = vectorizer.embed_many(
        contents=["chunk one", "chunk two", "chunk three"], input_type="document"
    )

    # One embedding per input, no collapsing.
    assert len(embeddings) == 3

    _, kwargs = client.contextualized_embed.call_args
    assert kwargs["inputs"] == ["chunk one", "chunk two", "chunk three"]
    assert kwargs["enable_auto_chunking"] is True
    assert kwargs["chunk_size"] == 32000
    # contextualized_embed does not accept truncation.
    assert "truncation" not in kwargs


def test_voyageai_context_query_disables_auto_chunking():
    """Query inputs must not enable auto-chunking (document-only per VoyageAI)."""
    vectorizer, client, _ = _build_voyage_vectorizer("voyage-context-4")

    embedding = vectorizer.embed(content="find similar docs", input_type="query")

    assert len(embedding) == 4
    _, kwargs = client.contextualized_embed.call_args
    assert kwargs["inputs"] == ["find similar docs"]
    assert kwargs["input_type"] == "query"
    assert kwargs["enable_auto_chunking"] is False
    # chunk_size is only valid alongside auto-chunking, so it must be omitted.
    assert "chunk_size" not in kwargs


def test_voyageai_context_default_input_type_enables_auto_chunking():
    """Omitted input_type must default to document + auto-chunking.

    SemanticRouter / SemanticCache call embed(_many) without input_type; a flat
    list[str] with no type is rejected by VoyageAI, so the default must resolve
    to a document (auto-chunking on) rather than passing input_type=None.
    """
    vectorizer, client, _ = _build_voyage_vectorizer("voyage-context-4")

    embeddings = vectorizer.embed_many(contents=["chunk one", "chunk two"])

    assert len(embeddings) == 2
    _, kwargs = client.contextualized_embed.call_args
    assert kwargs["inputs"] == ["chunk one", "chunk two"]
    assert kwargs["input_type"] == "document"
    assert kwargs["enable_auto_chunking"] is True
    assert kwargs["chunk_size"] == 32000


@pytest.mark.asyncio
async def test_voyageai_context_adefault_input_type_enables_auto_chunking():
    """Async: omitted input_type must default to document + auto-chunking."""
    vectorizer, _, aclient = _build_voyage_vectorizer("voyage-context-4")

    embeddings = await vectorizer.aembed_many(contents=["chunk one", "chunk two"])

    assert len(embeddings) == 2
    _, kwargs = aclient.contextualized_embed.call_args
    assert kwargs["input_type"] == "document"
    assert kwargs["enable_auto_chunking"] is True
    assert kwargs["chunk_size"] == 32000


@pytest.mark.asyncio
async def test_voyageai_context_aembed_many_uses_contextualized_api():
    """Async context models route to contextualized_embed with auto-chunking."""
    vectorizer, _, aclient = _build_voyage_vectorizer("voyage-context-4")

    embeddings = await vectorizer.aembed_many(
        contents=["chunk one", "chunk two"], input_type="document"
    )

    assert len(embeddings) == 2
    _, kwargs = aclient.contextualized_embed.call_args
    assert kwargs["inputs"] == ["chunk one", "chunk two"]
    assert kwargs["enable_auto_chunking"] is True
    assert kwargs["chunk_size"] == 32000


@pytest.mark.asyncio
async def test_voyageai_context_aquery_disables_auto_chunking():
    """Async query inputs must not enable auto-chunking."""
    vectorizer, _, aclient = _build_voyage_vectorizer("voyage-context-4")

    embedding = await vectorizer.aembed(content="find similar docs", input_type="query")

    assert len(embedding) == 4
    _, kwargs = aclient.contextualized_embed.call_args
    assert kwargs["inputs"] == ["find similar docs"]
    assert kwargs["input_type"] == "query"
    assert kwargs["enable_auto_chunking"] is False
    assert "chunk_size" not in kwargs


def test_voyageai_multimodal_embed_many_does_not_collapse_inputs():
    """Multimodal embed_many sends each content as its own input (no collapsing)."""
    vectorizer, client, _ = _build_voyage_vectorizer("voyage-multimodal-3.5")

    embeddings = vectorizer.embed_many(
        contents=["Ocean waves", "Forest trees"], input_type="document"
    )

    # Two requested contents must yield two embeddings.
    assert len(embeddings) == 2

    args, _ = client.multimodal_embed.call_args
    # Each item is wrapped as its own single-part multimodal input.
    assert args[0] == [["Ocean waves"], ["Forest trees"]]


# --- VoyageAI token-aware batching tests (mocked, no API key required) ---


def test_voyageai_token_aware_batching_splits_on_token_limit():
    """Plain text batches split when the per-request token budget is reached."""
    vectorizer, client, _ = _build_voyage_vectorizer("voyage-3-large")
    # 3 tokens per text; a 7-token budget fits two texts (6) but not three (9).
    vectorizer._token_limit = lambda: 7
    client.embed.reset_mock()

    texts = ["a a a", "b b b", "c c c"]
    embeddings = vectorizer.embed_many(contents=texts, input_type="document")

    assert len(embeddings) == 3
    batches = [call.args[0] for call in client.embed.call_args_list]
    assert batches == [["a a a", "b b b"], ["c c c"]]


def test_voyageai_token_aware_batching_oversized_text_goes_alone():
    """A single text over the token budget is still sent alone, not dropped."""
    vectorizer, client, _ = _build_voyage_vectorizer("voyage-3-large")
    vectorizer._token_limit = lambda: 5
    client.embed.reset_mock()

    texts = ["a a a a a a a a", "b b"]  # 8 tokens (> budget), then 2 tokens
    embeddings = vectorizer.embed_many(contents=texts, input_type="document")

    assert len(embeddings) == 2
    batches = [call.args[0] for call in client.embed.call_args_list]
    assert batches == [["a a a a a a a a"], ["b b"]]


def test_voyageai_token_aware_batching_respects_item_cap():
    """The per-model item cap still bounds a batch when tokens are plentiful."""
    vectorizer, client, _ = _build_voyage_vectorizer("voyage-4")  # item cap = 10
    vectorizer._token_limit = lambda: 10_000_000  # effectively unbounded
    client.embed.reset_mock()

    texts = ["x"] * 25  # 1 token each
    vectorizer.embed_many(contents=texts, input_type="document")

    batches = [call.args[0] for call in client.embed.call_args_list]
    assert [len(b) for b in batches] == [10, 10, 5]


@pytest.mark.asyncio
async def test_voyageai_token_aware_batching_async_splits_on_token_limit():
    """Async plain text batches split on the token budget too (sync/async parity)."""
    vectorizer, _, aclient = _build_voyage_vectorizer("voyage-3-large")
    vectorizer._token_limit = lambda: 7
    aclient.embed.reset_mock()

    texts = ["a a a", "b b b", "c c c"]
    embeddings = await vectorizer.aembed_many(contents=texts, input_type="document")

    assert len(embeddings) == 3
    batches = [call.args[0] for call in aclient.embed.call_args_list]
    assert batches == [["a a a", "b b b"], ["c c c"]]


def test_voyageai_tokenizer_unavailable_falls_back_to_item_batching():
    """If the tokenizer is unavailable, fall back to item-count batching.

    VoyageAI's tokenize() loads a HuggingFace tokenizer; when that can't be
    reached, embed_many must still succeed (item-count batches) instead of
    raising, and it must not re-attempt the failing tokenizer afterwards.
    """
    vectorizer, client, _ = _build_voyage_vectorizer("voyage-4")  # item cap = 10
    client.tokenize.side_effect = RuntimeError("HF hub unreachable")
    client.embed.reset_mock()

    texts = ["x"] * 25
    embeddings = vectorizer.embed_many(contents=texts, input_type="document")

    assert len(embeddings) == 25
    batches = [call.args[0] for call in client.embed.call_args_list]
    # Falls back to the per-model item cap (10), not token-aware sizing.
    assert [len(b) for b in batches] == [10, 10, 5]
    assert vectorizer._token_batching_supported is False

    # A second call must not re-invoke the failing tokenizer.
    client.tokenize.reset_mock()
    vectorizer.embed_many(contents=["y", "z"], input_type="document")
    client.tokenize.assert_not_called()


def test_voyageai_init_survives_tokenizer_unavailable():
    """Init (dimension probe) must not fail when the tokenizer is unavailable."""
    from unittest.mock import patch

    client, aclient = _fake_voyage_clients()
    client.tokenize.side_effect = RuntimeError("HF hub unreachable")
    aclient.tokenize.side_effect = RuntimeError("HF hub unreachable")
    with (
        patch("voyageai.Client", return_value=client),
        patch("voyageai.AsyncClient", return_value=aclient),
    ):
        vectorizer = VoyageAIVectorizer(
            model="voyage-3-large", api_config={"api_key": "test"}
        )

    assert vectorizer.dims == 4
    assert vectorizer._token_batching_supported is False


@pytest.mark.asyncio
async def test_voyageai_atokenizer_unavailable_falls_back_to_item_batching():
    """Async: tokenizer failure falls back to item-count batching (sync/async parity)."""
    vectorizer, client, aclient = _build_voyage_vectorizer("voyage-4")  # item cap = 10
    # Token counting uses the sync client's tokenize even on the async path.
    client.tokenize.side_effect = RuntimeError("HF hub unreachable")
    aclient.embed.reset_mock()

    texts = ["x"] * 25
    embeddings = await vectorizer.aembed_many(contents=texts, input_type="document")

    assert len(embeddings) == 25
    batches = [call.args[0] for call in aclient.embed.call_args_list]
    assert [len(b) for b in batches] == [10, 10, 5]
    assert vectorizer._token_batching_supported is False


@pytest.mark.parametrize(
    "model, expected_batch_size",
    [
        ("voyage-2", 72),
        ("voyage-4-lite", 30),
        ("voyage-3.5-lite", 30),
        ("voyage-4", 10),
        ("voyage-3.5", 10),
        ("voyage-4-large", 7),
        ("voyage-4-nano", 7),
        ("voyage-code-4", 7),
        ("voyage-context-4", 7),
        ("voyage-3-large", 7),
    ],
)
def test_voyageai_batch_size_for_current_models(model, expected_batch_size):
    """Current-generation models fall into batch-size tiers matching their token limits."""
    vectorizer, _, _ = _build_voyage_vectorizer(model)
    assert vectorizer._get_batch_size() == expected_batch_size
