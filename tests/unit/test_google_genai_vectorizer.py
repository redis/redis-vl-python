import builtins
import sys
import types

import pytest

from redisvl.utils.vectorize.base import BaseVectorizer
from redisvl.utils.vectorize.googlegenai import GoogleGenAIVectorizer

# Environment variables that would otherwise steer backend auto-detection.
# Scrubbed so each test sets only what it exercises, rather than inheriting
# GCP_* from a developer's shell or .env.
_GOOGLE_ENV_VARS = (
    "GCP_PROJECT_ID",
    "GCP_LOCATION",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


def _vec(content: str, dim):
    """Fake embedding: reduced-dim vectors are intentionally NON-normalized so
    tests can verify the vectorizer normalizes them."""
    if dim:
        return [float(i + 1) for i in range(dim)]
    base = float(len(content))
    return [base, base + 1.0, base + 2.0]


class FakeContentEmbedding:
    def __init__(self, values):
        self.values = values


class FakeEmbedResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeEmbedContentConfig:
    _allowed = {
        "task_type",
        "output_dimensionality",
        "title",
        "mime_type",
        "auto_truncate",
    }

    def __init__(self, **kwargs):
        unknown = set(kwargs) - self._allowed
        if unknown:
            raise TypeError(f"Unexpected EmbedContentConfig fields: {unknown}")
        for field in self._allowed:
            setattr(self, field, kwargs.get(field))


def _response_for(contents, config):
    dim = getattr(config, "output_dimensionality", None) if config else None
    items = [contents] if isinstance(contents, str) else list(contents)
    return FakeEmbedResponse([FakeContentEmbedding(_vec(c, dim)) for c in items])


class FakeModels:
    def __init__(self, client):
        self._client = client

    def embed_content(self, *, model, contents, config=None):
        self._client.calls.append(
            {"model": model, "contents": contents, "config": config, "async": False}
        )
        if self._client.raise_exc is not None:
            raise self._client.raise_exc
        return _response_for(contents, config)


class FakeAioModels:
    def __init__(self, client):
        self._client = client

    async def embed_content(self, *, model, contents, config=None):
        self._client.calls.append(
            {"model": model, "contents": contents, "config": config, "async": True}
        )
        if self._client.raise_exc is not None:
            raise self._client.raise_exc
        return _response_for(contents, config)


class FakeAio:
    def __init__(self, client):
        self.models = FakeAioModels(client)


class FakeClient:
    instances: list = []
    # Set on the class to force a failure during __init__'s dimension probe.
    raise_exc_on_init = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list = []
        self.raise_exc = self.__class__.raise_exc_on_init
        self.models = FakeModels(self)
        self.aio = FakeAio(self)
        FakeClient.instances.append(self)


@pytest.fixture(autouse=True)
def _scrub_google_env(monkeypatch):
    for var in _GOOGLE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_genai(monkeypatch):
    FakeClient.instances = []
    FakeClient.raise_exc_on_init = None

    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")
    types_mod.EmbedContentConfig = FakeEmbedContentConfig
    genai_mod.Client = FakeClient
    genai_mod.types = types_mod

    import google

    monkeypatch.setattr(google, "genai", genai_mod, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    return genai_mod


# --------------------------------------------------------------------------- #
# Backend resolution                                                          #
# --------------------------------------------------------------------------- #


def test_init_vertex_backend_from_api_config(fake_genai):
    vectorizer = GoogleGenAIVectorizer(
        api_config={"project_id": "proj", "location": "us-central1"}
    )
    client = FakeClient.instances[0]
    assert vectorizer.backend == "vertex"
    assert vectorizer.type == "google_genai"
    assert vectorizer.dims == 3
    assert client.kwargs["vertexai"] is True
    assert client.kwargs["project"] == "proj"
    assert client.kwargs["location"] == "us-central1"


def test_init_gemini_backend_from_api_config(fake_genai):
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "secret-key"})
    client = FakeClient.instances[0]
    assert vectorizer.backend == "gemini"
    assert client.kwargs.get("api_key") == "secret-key"
    assert "vertexai" not in client.kwargs


def test_explicit_backend_override_beats_creds(fake_genai):
    # project present, but explicitly force gemini
    vectorizer = GoogleGenAIVectorizer(
        api_config={"backend": "gemini", "project_id": "proj", "api_key": "k"}
    )
    assert vectorizer.backend == "gemini"


def test_explicit_api_key_beats_ambient_project(fake_genai, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "amb-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    assert vectorizer.backend == "gemini"


def test_env_project_selects_vertex(fake_genai, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    vectorizer = GoogleGenAIVectorizer()
    assert vectorizer.backend == "vertex"
    assert FakeClient.instances[0].kwargs["project"] == "p"


def test_env_key_selects_gemini(fake_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    vectorizer = GoogleGenAIVectorizer()
    assert vectorizer.backend == "gemini"


def test_both_env_creds_vertex_wins(fake_genai, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    vectorizer = GoogleGenAIVectorizer()
    assert vectorizer.backend == "vertex"


def test_legacy_gcp_env_vars_supported(fake_genai, monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "p")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")
    vectorizer = GoogleGenAIVectorizer()
    assert vectorizer.backend == "vertex"
    assert FakeClient.instances[0].kwargs["location"] == "us-central1"


def test_vertex_missing_location_raises(fake_genai):
    with pytest.raises(ValueError, match="location"):
        GoogleGenAIVectorizer(api_config={"project_id": "p"})


def test_no_credentials_raises(fake_genai):
    with pytest.raises(ValueError, match="Could not resolve a Google backend"):
        GoogleGenAIVectorizer()


def test_invalid_backend_value_raises(fake_genai):
    with pytest.raises(ValueError, match="Must be 'vertex' or 'gemini'"):
        GoogleGenAIVectorizer(api_config={"backend": "bogus"})


def test_api_config_not_mutated(fake_genai):
    api_config = {"api_key": "k"}
    GoogleGenAIVectorizer(api_config=api_config)
    assert api_config == {"api_key": "k"}


# --------------------------------------------------------------------------- #
# Embedding                                                                    #
# --------------------------------------------------------------------------- #


def test_embed_forwards_model_and_returns_values(fake_genai):
    vectorizer = GoogleGenAIVectorizer(
        model="gemini-embedding-001", api_config={"api_key": "k"}
    )
    result = vectorizer.embed("hello world")
    client = FakeClient.instances[0]
    assert result == _vec("hello world", None)
    assert client.calls[-1]["model"] == "gemini-embedding-001"
    assert client.calls[-1]["contents"] == "hello world"


def test_embed_many_batches_and_preserves_order(fake_genai):
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    result = vectorizer.embed_many(["a", "bb", "ccc", "dddd"], batch_size=2)
    assert result == [
        _vec("a", None),
        _vec("bb", None),
        _vec("ccc", None),
        _vec("dddd", None),
    ]
    client = FakeClient.instances[0]
    # calls[0] is the dimension probe; the rest are the batches.
    assert [c["contents"] for c in client.calls[1:]] == [["a", "bb"], ["ccc", "dddd"]]


def test_embed_many_empty_makes_no_api_calls(fake_genai):
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    client = FakeClient.instances[0]
    calls_after_init = len(client.calls)
    assert vectorizer.embed_many([]) == []
    assert len(client.calls) == calls_after_init


@pytest.mark.asyncio
async def test_aembed_uses_async_client(fake_genai):
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    result = await vectorizer.aembed("hello async")
    client = FakeClient.instances[0]
    assert result == _vec("hello async", None)
    assert client.calls[-1]["async"] is True


@pytest.mark.asyncio
async def test_aembed_many_uses_async_client(fake_genai):
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    result = await vectorizer.aembed_many(["a", "bb", "ccc"], batch_size=2)
    assert result == [_vec("a", None), _vec("bb", None), _vec("ccc", None)]
    client = FakeClient.instances[0]
    assert [c["contents"] for c in client.calls if c["async"]] == [
        ["a", "bb"],
        ["ccc"],
    ]


# --------------------------------------------------------------------------- #
# Dimensions, normalization, config passthrough                               #
# --------------------------------------------------------------------------- #


def test_output_dimensionality_sets_dims(fake_genai):
    vectorizer = GoogleGenAIVectorizer(
        api_config={"api_key": "k"}, output_dimensionality=6
    )
    assert vectorizer.dims == 6
    result = vectorizer.embed("hello")
    # Raw provider values are returned as-is (no normalization).
    assert result == _vec("hello", 6)


def test_per_call_output_dimensionality_is_rejected(fake_genai):
    # output_dimensionality determines dims, so it is fixed at construction and must
    # not be overridable per call (that would desync from the index vector width).
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    with pytest.raises(TypeError, match="output_dimensionality"):
        vectorizer.embed("hello", output_dimensionality=5)


def test_task_type_passthrough_per_call(fake_genai):
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    vectorizer.embed("hello", task_type="RETRIEVAL_QUERY")
    config = FakeClient.instances[0].calls[-1]["config"]
    assert config.task_type == "RETRIEVAL_QUERY"


def test_task_type_default_applied_to_every_call(fake_genai):
    vectorizer = GoogleGenAIVectorizer(
        api_config={"api_key": "k"}, task_type="RETRIEVAL_DOCUMENT"
    )
    vectorizer.embed("hello")
    config = FakeClient.instances[0].calls[-1]["config"]
    assert config.task_type == "RETRIEVAL_DOCUMENT"


# --------------------------------------------------------------------------- #
# Input validation and errors                                                  #
# --------------------------------------------------------------------------- #


def test_rejects_invalid_single_content(fake_genai):
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    with pytest.raises(TypeError):
        vectorizer.embed(42)


def test_rejects_invalid_many_contents(fake_genai):
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    with pytest.raises(TypeError):
        vectorizer.embed_many("not a list")
    with pytest.raises(TypeError):
        vectorizer.embed_many(["valid", 42])


def test_invalid_dtype_uses_base_validation(fake_genai):
    with pytest.raises(ValueError, match="Invalid data type"):
        GoogleGenAIVectorizer(api_config={"api_key": "k"}, dtype="float25")


def test_missing_dependency_raises_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google" and "genai" in (fromlist or ()):
            raise ImportError("no genai")
        if name == "google.genai":
            raise ImportError("no genai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    monkeypatch.delitem(sys.modules, "google.genai.types", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    with pytest.raises(ImportError, match=r"pip install redisvl\[google-genai\]"):
        GoogleGenAIVectorizer()


def test_transient_error_is_retried(fake_genai, monkeypatch):
    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    monkeypatch.setattr(GoogleGenAIVectorizer._embed.retry, "sleep", lambda _: None)
    client = FakeClient.instances[0]
    client.raise_exc = RuntimeError("transient 503")
    calls_before = len(client.calls)

    with pytest.raises(RuntimeError, match="transient 503"):
        vectorizer.embed("hello")

    assert len(client.calls) - calls_before == 6


@pytest.mark.asyncio
async def test_async_transient_error_is_retried(fake_genai, monkeypatch):
    async def _no_sleep(_):
        return None

    vectorizer = GoogleGenAIVectorizer(api_config={"api_key": "k"})
    monkeypatch.setattr(GoogleGenAIVectorizer._aembed.retry, "sleep", _no_sleep)
    client = FakeClient.instances[0]
    client.raise_exc = RuntimeError("transient 503")
    calls_before = len(client.calls)

    with pytest.raises(RuntimeError, match="transient 503"):
        await vectorizer.aembed("hello")

    assert len(client.calls) - calls_before == 6


def test_credentials_never_appear_in_raised_error(fake_genai, monkeypatch):
    # Force the dimension probe to fail with an error that echoes the api key.
    secret = "super-secret-key-12345"
    monkeypatch.setattr(
        FakeClient,
        "raise_exc_on_init",
        RuntimeError(f"HTTP 401 for key={secret}"),
    )
    with pytest.raises(ValueError) as exc_info:
        GoogleGenAIVectorizer(api_config={"api_key": secret})
    assert secret not in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Wiring                                                                        #
# --------------------------------------------------------------------------- #


def test_vectorizer_from_dict_supports_google_genai(fake_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    from redisvl.utils.vectorize import vectorizer_from_dict

    vectorizer = vectorizer_from_dict(
        {"type": "google_genai", "model": "gemini-embedding-001", "dtype": "float64"}
    )
    assert isinstance(vectorizer, GoogleGenAIVectorizer)
    assert vectorizer.model == "gemini-embedding-001"
    assert vectorizer.dtype == "float64"
    assert vectorizer.backend == "gemini"


def test_enum_and_public_export():
    from redisvl.utils.vectorize.base import Vectorizers

    assert Vectorizers("google_genai") == Vectorizers.google_genai

    import redisvl.utils.vectorize as vectorize

    assert vectorize.GoogleGenAIVectorizer is GoogleGenAIVectorizer


def test_uses_base_public_batch_embedding_methods():
    assert GoogleGenAIVectorizer.embed_many is BaseVectorizer.embed_many
    assert GoogleGenAIVectorizer.aembed_many is BaseVectorizer.aembed_many
