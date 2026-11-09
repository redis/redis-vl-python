"""Credentials missing from a non-empty ``api_config`` must fall back to env vars."""

import importlib
import sys
import types

import pytest

CALLS: list = []


class _Recorder:
    def __init__(self, **kwargs):
        CALLS.append(kwargs)

    def embed(self, *args, **kwargs):  # bound by the VoyageAI vectorizer
        return None

    multimodal_embed = contextualized_embed = embed


def _fake_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


# (id, module, class, env var, config key, fake package)
CASES = [
    (
        "cohere-vec",
        "redisvl.utils.vectorize.text.cohere",
        "CohereTextVectorizer",
        "COHERE_API_KEY",
        "api_key",
        "cohere",
    ),
    (
        "mistral-vec",
        "redisvl.utils.vectorize.text.mistral",
        "MistralAITextVectorizer",
        "MISTRAL_API_KEY",
        "api_key",
        "mistralai",
    ),
    (
        "voyage-vec",
        "redisvl.utils.vectorize.voyageai",
        "VoyageAIVectorizer",
        "VOYAGE_API_KEY",
        "api_key",
        "voyageai",
    ),
    (
        "cohere-rerank",
        "redisvl.utils.rerank.cohere",
        "CohereReranker",
        "COHERE_API_KEY",
        "api_key",
        "cohere",
    ),
    (
        "voyage-rerank",
        "redisvl.utils.rerank.voyageai",
        "VoyageAIReranker",
        "VOYAGE_API_KEY",
        "api_key",
        "voyageai",
    ),
]


@pytest.fixture(autouse=True)
def fake_clients(monkeypatch):
    CALLS.clear()
    for name in (
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "VOYAGE_API_KEY",
        "GCP_PROJECT_ID",
        "GCP_LOCATION",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in ("cohere", "voyageai"):
        monkeypatch.setitem(
            sys.modules,
            name,
            _fake_module(name, Client=_Recorder, AsyncClient=_Recorder),
        )
    monkeypatch.setitem(
        sys.modules, "mistralai", _fake_module("mistralai", Mistral=_Recorder)
    )
    monkeypatch.setitem(
        sys.modules,
        "vertexai",
        _fake_module("vertexai", init=lambda **kwargs: CALLS.append(kwargs)),
    )


def _build(module_name, class_name, monkeypatch, **kwargs):
    cls = getattr(importlib.import_module(module_name), class_name)
    if "rerank" in module_name:
        kwargs.setdefault("model", "fake-rerank-model")
    if hasattr(cls, "_set_model_dims"):
        monkeypatch.setattr(cls, "_set_model_dims", lambda self: 3)
    return cls(**kwargs)


@pytest.mark.parametrize(
    "module_name,class_name,env,key,_pkg",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
@pytest.mark.parametrize("api_config", [None, {}, {"unrelated": 1}, {"api_key": ""}])
def test_missing_or_empty_api_key_falls_back_to_env(
    monkeypatch, module_name, class_name, env, key, _pkg, api_config
):
    monkeypatch.setenv(env, "env-key")
    _build(module_name, class_name, monkeypatch, api_config=api_config)
    assert CALLS and all(call["api_key"] == "env-key" for call in CALLS)


@pytest.mark.parametrize(
    "module_name,class_name,env,key,_pkg",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_api_config_key_wins_over_env(
    monkeypatch, module_name, class_name, env, key, _pkg
):
    monkeypatch.setenv(env, "env-key")
    _build(module_name, class_name, monkeypatch, api_config={key: "cfg-key"})
    assert CALLS and all(call["api_key"] == "cfg-key" for call in CALLS)


@pytest.mark.parametrize(
    "module_name,class_name,env,key,_pkg",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_no_key_anywhere_still_raises(
    monkeypatch, module_name, class_name, env, key, _pkg
):
    with pytest.raises(ValueError, match="API key is required"):
        _build(module_name, class_name, monkeypatch, api_config={"unrelated": 1})


def test_vertexai_project_and_location_fall_back_to_env(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "env-project")
    monkeypatch.setenv("GCP_LOCATION", "env-location")
    _build(
        "redisvl.utils.vectorize.vertexai",
        "VertexAIVectorizer",
        monkeypatch,
        api_config={"credentials": "creds"},
    )
    assert CALLS == [
        {"project": "env-project", "location": "env-location", "credentials": "creds"}
    ]


def test_vertexai_config_wins_over_env(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "env-project")
    monkeypatch.setenv("GCP_LOCATION", "env-location")
    _build(
        "redisvl.utils.vectorize.vertexai",
        "VertexAIVectorizer",
        monkeypatch,
        api_config={"project_id": "cfg-project", "location": "cfg-location"},
    )
    assert CALLS[0]["project"] == "cfg-project"
    assert CALLS[0]["location"] == "cfg-location"
