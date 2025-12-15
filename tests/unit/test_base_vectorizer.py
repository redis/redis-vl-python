from typing import List

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

        def embed(self, content: str, **kwargs) -> List[float]:
            return [0.0] * self.dims

        async def aembed(self, content: str, **kwargs) -> List[float]:
            return [0.0] * self.dims

        async def aembed_many(self, contents: List[str], **kwargs) -> List[List[float]]:
            return [[0.0] * self.dims] * len(contents)

        def embed_many(self, contents: List[str], **kwargs) -> List[List[float]]:
            return [[0.0] * self.dims] * len(contents)

    vectorizer = SimpleVectorizer()
    assert vectorizer.model == "simple"
    assert vectorizer.dims == 10
    assert vectorizer.dtype == "float32"
