import os

import pytest

from redisvl.utils.vectorize import (
    AzureOpenAITextVectorizer,
    CohereTextVectorizer,
    CustomTextVectorizer,
    HFTextVectorizer,
    MistralAITextVectorizer,
    OpenAITextVectorizer,
    VertexAITextVectorizer,
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
        MistralAITextVectorizer,
        CustomTextVectorizer,
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
    elif request.param == MistralAITextVectorizer:
        return request.param()
    elif request.param == AzureOpenAITextVectorizer:
        return request.param(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-ada-002")
        )
    elif request.param == CustomTextVectorizer:

        def embed(text):
            return [1.1, 2.2, 3.3, 4.4]

        def embed_many(texts):
            return [[1.1, 2.2, 3.3, 4.4]] * len(texts)

        return request.param(embed=embed, embed_many=embed_many)


@pytest.fixture
def custom_embed_func():
    def embed(text: str):
        return [1.1, 2.2, 3.3, 4.4]

    return embed


@pytest.fixture
def custom_embed_class():
    class embedder:
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

    return embedder


def test_vectorizer_embed(vectorizer):
    text = "This is a test sentence."
    if isinstance(vectorizer, CohereTextVectorizer):
        embedding = vectorizer.embed(text, input_type="search_document")
    else:
        embedding = vectorizer.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == vectorizer.dims


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


def test_vectorizer_bad_input(vectorizer):
    with pytest.raises(TypeError):
        vectorizer.embed(1)

    with pytest.raises(TypeError):
        vectorizer.embed({"foo": "bar"})

    with pytest.raises(TypeError):
        vectorizer.embed_many(42)


def test_custom_vectorizer_embed(custom_embed_class, custom_embed_func):
    # test we can pass a stand alone function as embedder callable
    custom_wrapper = CustomTextVectorizer(embed=custom_embed_func)
    embedding = custom_wrapper.embed("This is a test sentence.")
    assert embedding == [1.1, 2.2, 3.3, 4.4]

    # test we can pass an instance of a class method as embedder callable
    custom_wrapper = CustomTextVectorizer(embed=custom_embed_class().embed)
    embedding = custom_wrapper.embed("This is a test sentence.")
    assert embedding == [1.1, 2.2, 3.3, 4.4]

    # test we can pass additional parameters and kwargs to embedding methods
    custom_wrapper = CustomTextVectorizer(embed=custom_embed_class().embed_with_args)
    embedding = custom_wrapper.embed("This is a test sentence.", max_len=4)
    assert embedding == [1.1, 2.2, 3.3, 4.4]
    embedding = custom_wrapper.embed("This is a test sentence.", max_len=2)
    assert embedding == [1.1, 2.2]

    # test that correct error is raised if a non-callable is passed
    with pytest.raises(TypeError):
        bad_wrapper = CustomTextVectorizer(embed="hello")

    with pytest.raises(TypeError):
        bad_wrapper = CustomTextVectorizer(embed=42)

    with pytest.raises(TypeError):
        bad_wrapper = CustomTextVectorizer(embed={"foo": "bar"})

    # test that correct error is raised if passed function has incorrect types
    def bad_arg_type(value: int):
        return [value]

    with pytest.raises(ValueError):
        bad_wrapper = CustomTextVectorizer(embed=bad_arg_type)

    def bad_return_type(text: str) -> str:
        return text

    with pytest.raises(ValueError):
        bad_wrapper = CustomTextVectorizer(embed=bad_return_type)


def test_custom_vectorizer_embed_many(custom_embed_class, custom_embed_func):
    # test we can pass a stand alone function as embed_many callable
    custom_wrapper = CustomTextVectorizer(
        custom_embed_func, embed_many=custom_embed_class().embed_many
    )
    embeddings = custom_wrapper.embed_many(["test one.", "test two"])
    assert embeddings == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    # test we can pass a class method as embedder callable
    custom_wrapper = CustomTextVectorizer(
        custom_embed_func, embed_many=custom_embed_class().embed_many
    )
    embeddings = custom_wrapper.embed_many(["test one.", "test two"])
    assert embeddings == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    # test we can pass additional parameters and kwargs to embedding methods
    custom_wrapper = CustomTextVectorizer(
        custom_embed_func, embed_many=custom_embed_class().embed_many_with_args
    )
    embeddings = custom_wrapper.embed_many(["test one.", "test two"], param=True)
    assert embeddings == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    embeddings = custom_wrapper.embed_many(["test one.", "test two"], param=False)
    assert embeddings == [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]

    # test that correct error is raised if a non-callable is passed
    with pytest.raises(TypeError):
        bad_wrapper = CustomTextVectorizer(custom_embed_func, embed_many="hello")

    with pytest.raises(TypeError):
        bad_wrapper = CustomTextVectorizer(custom_embed_func, embed_many=42)

    with pytest.raises(TypeError):
        bad_wrapper = CustomTextVectorizer(custom_embed_func, embed_many={"foo": "bar"})

    # test that correct error is raised if passed function has incorrect types
    def bad_arg_type(value: int):
        return [value]

    with pytest.raises(ValueError):
        bad_wrapper = CustomTextVectorizer(custom_embed_func, embed_many=bad_arg_type)

    def bad_return_type(text: str) -> str:
        return text

    with pytest.raises(ValueError):
        bad_wrapper = CustomTextVectorizer(
            custom_embed_func, embed_many=bad_return_type
        )


@pytest.fixture(
    params=[OpenAITextVectorizer, MistralAITextVectorizer, CustomTextVectorizer]
)
def avectorizer(request, skip_vectorizer):
    if skip_vectorizer:
        pytest.skip("Skipping vectorizer instantiation...")

    # Here we use actual models for integration test
    if request.param == OpenAITextVectorizer:
        return request.param()
    elif request.param == MistralAITextVectorizer:
        return request.param()

    # Here we use actual models for integration test
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
