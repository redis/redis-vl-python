import sys
import types

import pytest

from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer


class _Item:
    def __init__(self, embedding):
        self.embedding = embedding


class _Response:
    def __init__(self, data):
        self.data = data


class _Embeddings:
    def create(self, *, input, model, **kwargs):
        return _Response([_Item([1.0, 2.0, 3.0]) for _ in input])


class FakeOpenAI:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.embeddings = _Embeddings()
        FakeOpenAI.instances.append(self)


class FakeAsyncOpenAI:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeAsyncOpenAI.instances.append(self)


@pytest.fixture(autouse=True)
def fake_openai(monkeypatch):
    FakeOpenAI.instances = []
    FakeAsyncOpenAI.instances = []
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    module.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_api_config_without_api_key_falls_back_to_env(monkeypatch):
    """A custom base_url must not make OPENAI_API_KEY unreachable."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    OpenAITextVectorizer(api_config={"base_url": "https://proxy.example.test/v1"})

    for client in (FakeOpenAI.instances[0], FakeAsyncOpenAI.instances[0]):
        assert client.kwargs["api_key"] == "env-key"
        assert client.kwargs["base_url"] == "https://proxy.example.test/v1"


def test_api_config_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    OpenAITextVectorizer(api_config={"api_key": "cfg-key"})

    assert FakeOpenAI.instances[0].kwargs["api_key"] == "cfg-key"
    assert FakeAsyncOpenAI.instances[0].kwargs["api_key"] == "cfg-key"


def test_api_config_not_mutated():
    config = {"api_key": "cfg-key", "base_url": "https://proxy.example.test/v1"}
    OpenAITextVectorizer(api_config=config)

    assert config == {
        "api_key": "cfg-key",
        "base_url": "https://proxy.example.test/v1",
    }


def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        OpenAITextVectorizer(api_config={"base_url": "https://proxy.example.test/v1"})
