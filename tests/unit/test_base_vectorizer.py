from typing import Any

from redisvl.utils.vectorize.base import BaseVectorizer


def test_base_vectorizer_defaults():
    """
    Test that the base vectorizer defaults are set correctly, with
    a default for dtype. Versions before 0.3.8 did not have this field.

    A regression test for langchain-redis/#48
    """

    class SimpleVectorizer(BaseVectorizer):
        model: str = "simple"
        dims: int = 10

        def embed(self, content: Any = "", text: Any = "", **kwargs) -> list[float]:
            return [0.0] * self.dims

        async def aembed(
            self, content: Any = "", text: Any = "", **kwargs
        ) -> list[float]:
            return [0.0] * self.dims

        async def aembed_many(
            self,
            contents: list[Any] | None = None,
            texts: list[Any] | None = None,
            **kwargs,
        ) -> list[list[float]]:
            return [[0.0] * self.dims] * len(contents)

        def embed_many(
            self,
            contents: list[Any] | None = None,
            texts: list[Any] | None = None,
            **kwargs,
        ) -> list[list[float]]:
            return [[0.0] * self.dims] * len(contents)

    vectorizer = SimpleVectorizer()
    assert vectorizer.model == "simple"
    assert vectorizer.dims == 10
    assert vectorizer.dtype == "float32"


class _InputTypeVectorizer(BaseVectorizer):
    """Fake provider whose output depends on the ``input_type`` kwarg."""

    model: str = "fake-model"
    dims: int = 1

    def _value(self, kwargs: dict) -> float:
        return 2.0 if kwargs.get("input_type") == "search_document" else 1.0

    def _embed(self, content: Any, **kwargs) -> list[float]:
        return [self._value(kwargs)]

    def _embed_many(
        self, contents: list[Any], batch_size: int = 10, **kwargs
    ) -> list[list[float]]:
        return [[self._value(kwargs)] for _ in contents]

    async def _aembed(self, content: Any, **kwargs) -> list[float]:
        return [self._value(kwargs)]

    async def _aembed_many(
        self, contents: list[Any], batch_size: int = 10, **kwargs
    ) -> list[list[float]]:
        return [[self._value(kwargs)] for _ in contents]


class _FakeCache:
    """In-memory stand-in for EmbeddingsCache keyed by (content, model_name)."""

    def __init__(self):
        self.store: dict = {}

    def get(self, content, model_name):
        return self.store.get((content, model_name))

    def set(self, content, model_name, embedding, metadata=None):
        self.store[(content, model_name)] = {"embedding": embedding}

    def mget(self, contents, model_name):
        return [self.get(c, model_name) for c in contents]

    def mset(self, items):
        for i in items:
            self.set(i["content"], i["model_name"], i["embedding"])

    async def aget(self, content, model_name):
        return self.get(content, model_name)

    async def aset(self, content, model_name, embedding, metadata=None):
        self.set(content, model_name, embedding, metadata)

    async def amget(self, contents, model_name):
        return self.mget(contents, model_name)

    async def amset(self, items):
        self.mset(items)


def _cached_vectorizer() -> _InputTypeVectorizer:
    vec = _InputTypeVectorizer()
    vec.cache = _FakeCache()
    return vec


def test_embed_cache_distinguishes_request_kwargs():
    vec = _cached_vectorizer()
    assert vec.embed("x", input_type="search_query") == [1.0]
    assert vec.embed("x", input_type="search_document") == [2.0]
    # Same kwargs hit the cache
    assert vec.embed("x", input_type="search_document") == [2.0]


def test_embed_many_cache_distinguishes_request_kwargs():
    vec = _cached_vectorizer()
    assert vec.embed_many(["x"], input_type="search_query") == [[1.0]]
    assert vec.embed_many(["x"], input_type="search_document") == [[2.0]]


async def test_aembed_cache_distinguishes_request_kwargs():
    vec = _cached_vectorizer()
    assert await vec.aembed("x", input_type="search_query") == [1.0]
    assert await vec.aembed("x", input_type="search_document") == [2.0]


async def test_aembed_many_cache_distinguishes_request_kwargs():
    vec = _cached_vectorizer()
    assert await vec.aembed_many(["x"], input_type="search_query") == [[1.0]]
    assert await vec.aembed_many(["x"], input_type="search_document") == [[2.0]]


def test_embed_cache_key_without_kwargs_is_unchanged():
    vec = _cached_vectorizer()
    vec.embed("x", metadata={"a": 1})
    vec.embed_many(["y"])
    assert {k[1] for k in vec.cache.store} == {"fake-model"}
