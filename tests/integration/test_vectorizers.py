import os

import pytest

from redisvl.utils.vectorize import (
    AzureOpenAITextVectorizer,
    BedrockTextVectorizer,
    CohereTextVectorizer,
    CustomTextVectorizer,
    HFTextVectorizer,
    MistralAITextVectorizer,
    OpenAITextVectorizer,
    VertexAITextVectorizer,
    VoyageAITextVectorizer,
)


@pytest.fixture(
    params=[
        HFTextVectorizer,
        OpenAITextVectorizer,
        VertexAITextVectorizer,
        CohereTextVectorizer,
        AzureOpenAITextVectorizer,
        BedrockTextVectorizer,
        MistralAITextVectorizer,
        CustomTextVectorizer,
        VoyageAITextVectorizer,
    ]
)
def vectorizer(request):
    if request.param == HFTextVectorizer:
        return request.param()
    elif request.param == OpenAITextVectorizer:
        return request.param()
    elif request.param == VertexAITextVectorizer:
        return request.param()
    elif request.param == CohereTextVectorizer:
        return request.param()
    elif request.param == MistralAITextVectorizer:
        return request.param()
    elif request.param == VoyageAITextVectorizer:
        return request.param(model="voyage-large-2")
    elif request.param == AzureOpenAITextVectorizer:
        return request.param(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-ada-002")
        )
    elif request.param == BedrockTextVectorizer:
        return request.param(
            model=os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0")
        )
    elif request.param == CustomTextVectorizer:

        def embed(text):
            return [1.1, 2.2, 3.3, 4.4]

        def embed_many(texts):
            return [[1.1, 2.2, 3.3, 4.4]] * len(texts)

        return request.param(embed=embed, embed_many=embed_many)


@pytest.fixture
def bedrock_vectorizer():
    return BedrockTextVectorizer(
        model=os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0")
    )


@pytest.fixture
def custom_embed_func():
    def embed(text: str):
        return [1.1, 2.2, 3.3, 4.4]

    return embed


@pytest.fixture
def custom_embed_class():
    class MyEmbedder:
        def embed(self, text: str):
            return [1.1, 2.2, 3.3, 4.4]

        def embed_with_args(self, text: str, max_len=None):
            return [1.1, 2.2, 3.3, 4.4][0:max_len]

        def embed_many(self, text_list):
            return [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

        def embed_many_with_args(self, texts, param=True):
            if param:
                return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            else:
                return [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]

    return MyEmbedder


@pytest.mark.requires_api_keys
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


@pytest.mark.requires_api_keys
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


@pytest.mark.requires_api_keys
def test_vectorizer_bad_input(vectorizer):
    with pytest.raises(TypeError):
        vectorizer.embed(1)

    with pytest.raises(TypeError):
        vectorizer.embed({"foo": "bar"})

    with pytest.raises(TypeError):
        vectorizer.embed_many(42)


@pytest.mark.requires_api_keys
def test_bedrock_bad_credentials():
    with pytest.raises(ValueError):
        BedrockTextVectorizer(
            api_config={
                "aws_access_key_id": "invalid",
                "aws_secret_access_key": "invalid",
            }
        )


@pytest.mark.requires_api_keys
def test_bedrock_invalid_model(bedrock_vectorizer):
    with pytest.raises(ValueError):
        bedrock = BedrockTextVectorizer(model="invalid-model")
        bedrock.embed("test")


def test_custom_vectorizer_embed(custom_embed_class, custom_embed_func):
    custom_wrapper = CustomTextVectorizer(embed=custom_embed_func)
    embedding = custom_wrapper.embed("This is a test sentence.")
    assert embedding == [1.1, 2.2, 3.3, 4.4]

    custom_wrapper = CustomTextVectorizer(embed=custom_embed_class().embed)
    embedding = custom_wrapper.embed("This is a test sentence.")
    assert embedding == [1.1, 2.2, 3.3, 4.4]

    custom_wrapper = CustomTextVectorizer(embed=custom_embed_class().embed_with_args)
    embedding = custom_wrapper.embed("This is a test sentence.", max_len=4)
    assert embedding == [1.1, 2.2, 3.3, 4.4]
    embedding = custom_wrapper.embed("This is a test sentence.", max_len=2)
    assert embedding == [1.1, 2.2]

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(embed="hello")

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(embed=42)

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(embed={"foo": "bar"})

    def bad_arg_type(value: int):
        return [value]

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(embed=bad_arg_type)

    def bad_return_type(text: str) -> str:
        return text

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(embed=bad_return_type)


def test_custom_vectorizer_embed_many(custom_embed_class, custom_embed_func):
    custom_wrapper = CustomTextVectorizer(
        custom_embed_func, embed_many=custom_embed_class().embed_many
    )
    embeddings = custom_wrapper.embed_many(["test one.", "test two"])
    assert embeddings == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    custom_wrapper = CustomTextVectorizer(
        custom_embed_func, embed_many=custom_embed_class().embed_many
    )
    embeddings = custom_wrapper.embed_many(["test one.", "test two"])
    assert embeddings == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    custom_wrapper = CustomTextVectorizer(
        custom_embed_func, embed_many=custom_embed_class().embed_many_with_args
    )
    embeddings = custom_wrapper.embed_many(["test one.", "test two"], param=True)
    assert embeddings == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    embeddings = custom_wrapper.embed_many(["test one.", "test two"], param=False)
    assert embeddings == [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(custom_embed_func, embed_many="hello")

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(custom_embed_func, embed_many=42)

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(
            custom_embed_func, embed_many={"foo": "bar"}
        )

    def bad_arg_type(value: int):
        return [value]

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(
            custom_embed_func, embed_many=bad_arg_type
        )

    def bad_return_type(text: str) -> str:
        return text

    with pytest.raises(ValueError):
        invalid_vectorizer = CustomTextVectorizer(
            custom_embed_func, embed_many=bad_return_type
        )


@pytest.mark.requires_api_keys
@pytest.mark.parametrize(
    "vectorizer_",
    [
        AzureOpenAITextVectorizer,
        BedrockTextVectorizer,
        CohereTextVectorizer,
        CustomTextVectorizer,
        HFTextVectorizer,
        MistralAITextVectorizer,
        OpenAITextVectorizer,
        VertexAITextVectorizer,
        VoyageAITextVectorizer,
    ],
)
def test_default_dtype(vectorizer_):
    # test dtype defaults to float32
    if issubclass(vectorizer_, CustomTextVectorizer):
        vectorizer = vectorizer_(embed=lambda x, input_type=None: [1.0, 2.0, 3.0])
    elif issubclass(vectorizer_, AzureOpenAITextVectorizer):
        vectorizer = vectorizer_(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-ada-002")
        )
    else:
        vectorizer = vectorizer_()

    assert vectorizer.dtype == "float32"


@pytest.mark.requires_api_keys
@pytest.mark.parametrize(
    "vectorizer_",
    [
        AzureOpenAITextVectorizer,
        BedrockTextVectorizer,
        CohereTextVectorizer,
        CustomTextVectorizer,
        HFTextVectorizer,
        MistralAITextVectorizer,
        OpenAITextVectorizer,
        VertexAITextVectorizer,
        VoyageAITextVectorizer,
    ],
)
def test_other_dtypes(vectorizer_):
    # test initializing dtype in constructor
    for dtype in ["float16", "float32", "float64", "bfloat16"]:
        if issubclass(vectorizer_, CustomTextVectorizer):
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


@pytest.mark.requires_api_keys
@pytest.mark.parametrize(
    "vectorizer_",
    [
        AzureOpenAITextVectorizer,
        BedrockTextVectorizer,
        CohereTextVectorizer,
        HFTextVectorizer,
        MistralAITextVectorizer,
        OpenAITextVectorizer,
        VertexAITextVectorizer,
        VoyageAITextVectorizer,
    ],
)
def test_bad_dtypes(vectorizer_):
    with pytest.raises(ValueError):
        vectorizer_(dtype="float25")

    with pytest.raises(ValueError):
        vectorizer_(dtype=7)

    with pytest.raises(ValueError):
        vectorizer_(dtype=None)


@pytest.fixture(
    params=[
        OpenAITextVectorizer,
        BedrockTextVectorizer,
        MistralAITextVectorizer,
        CustomTextVectorizer,
        VoyageAITextVectorizer,
    ]
)
def avectorizer(request):
    if request.param == CustomTextVectorizer:

        def embed_func(text):
            return [1.1, 2.2, 3.3, 4.4]

        async def aembed_func(text):
            return [1.1, 2.2, 3.3, 4.4]

        async def aembed_many_func(texts):
            return [[1.1, 2.2, 3.3, 4.4]] * len(texts)

        return request.param(
            embed=embed_func, aembed=aembed_func, aembed_many=aembed_many_func
        )
    else:
        return request.param()


@pytest.mark.requires_api_keys
@pytest.mark.asyncio
async def test_vectorizer_aembed(avectorizer):
    text = "This is a test sentence."
    embedding = await avectorizer.aembed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == avectorizer.dims


@pytest.mark.requires_api_keys
@pytest.mark.asyncio
async def test_vectorizer_aembed_many(avectorizer):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    embeddings = await avectorizer.aembed_many(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(
        isinstance(emb, list) and len(emb) == avectorizer.dims for emb in embeddings
    )


@pytest.mark.requires_api_keys
@pytest.mark.asyncio
async def test_avectorizer_bad_input(avectorizer):
    with pytest.raises(TypeError):
        avectorizer.embed(1)

    with pytest.raises(TypeError):
        avectorizer.embed({"foo": "bar"})

    with pytest.raises(TypeError):
        avectorizer.embed_many(42)
