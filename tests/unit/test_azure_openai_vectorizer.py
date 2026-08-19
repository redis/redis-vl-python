import builtins
import inspect
import sys
import types

import pytest

from redisvl.utils.vectorize.base import BaseVectorizer
from redisvl.utils.vectorize.text.azureopenai import AzureOpenAITextVectorizer

# Environment variables the vectorizer falls back to, per key, when that key is
# absent from api_config.
_AZURE_ENV_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
)

# Obviously-fake sentinels. The endpoint uses the RFC 2606 reserved .test TLD --
# NOT *.openai.azure.com, which resolves -- so if fake injection ever breaks, the
# call fails closed instead of reaching a live resource. Nothing here is shaped
# like a real credential.
ENDPOINT, API_VERSION, API_KEY = (
    "https://fake-resource.openai.azure.test",
    "2024-02-01-fake",
    "not-a-real-key",
)


def _creds():
    """A factory, not a module constant: _initialize_clients pops credentials out of
    the dict it is given. It pops from a copy (see test_api_config_not_mutated), but
    a shared constant would still be one regression away from being emptied by the
    first test and taking every later test down with it."""
    return {
        "azure_endpoint": ENDPOINT,
        "api_version": API_VERSION,
        "api_key": API_KEY,
    }


def _vec(content: str) -> list[float]:
    """Deterministic fake embedding, three dims so `dims == 3` after the probe."""
    base = float(len(content))
    return [base, base + 1.0, base + 2.0]


class FakeEmbeddingItem:
    """Mirrors one element of ``result.data``."""

    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingsResponse:
    """Mirrors the object returned by ``client.embeddings.create``."""

    def __init__(self, data):
        self.data = data


# One shared call log across both clients: assertions on ordering (probe first,
# then batches) and on which client served a call need a single timeline. The
# "async" flag is what distinguishes _client from _aclient.
_CALLS: list = []


class _FakeEmbeddings:
    """Provides ``client.embeddings.create`` for the sync client."""

    def __init__(self, client):
        self._client = client

    def create(self, *, input, model, **kwargs):
        _CALLS.append(
            {"input": list(input), "model": model, "kwargs": kwargs, "async": False}
        )
        if self._client.raise_exc is not None:
            raise self._client.raise_exc
        return FakeEmbeddingsResponse([FakeEmbeddingItem(_vec(c)) for c in input])


class _FakeAsyncEmbeddings:
    """Provides ``await client.embeddings.create`` for the async client."""

    def __init__(self, client):
        self._client = client

    async def create(self, *, input, model, **kwargs):
        _CALLS.append(
            {"input": list(input), "model": model, "kwargs": kwargs, "async": True}
        )
        if self._client.raise_exc is not None:
            raise self._client.raise_exc
        return FakeEmbeddingsResponse([FakeEmbeddingItem(_vec(c)) for c in input])


class FakeAzureOpenAI:
    # Deliberately NOT shared with FakeAsyncAzureOpenAI: a shared list would make
    # instances[-1] silently return the wrong client and tests would assert nothing.
    instances: list = []
    # Set on the class to force a failure during __init__'s dimension probe. There
    # is no instance to arm yet at that point; use `raise_exc` on the instance for
    # post-construction failures.
    raise_exc_on_init = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.raise_exc = self.__class__.raise_exc_on_init
        self.embeddings = _FakeEmbeddings(self)
        FakeAzureOpenAI.instances.append(self)


class FakeAsyncAzureOpenAI:
    instances: list = []
    raise_exc_on_init = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.raise_exc = self.__class__.raise_exc_on_init
        self.embeddings = _FakeAsyncEmbeddings(self)
        FakeAsyncAzureOpenAI.instances.append(self)


@pytest.fixture(autouse=True)
def _scrub_azure_env(monkeypatch):
    """Hermeticity, and that alone. ``os.environ`` is process-global, and several
    tests below assert on what happens when a credential is *absent*. A developer
    with real ``AZURE_OPENAI_*`` vars exported would satisfy those reads,
    construction would succeed, and those tests would fail loudly -- not pass
    vacuously. So this fixture is what makes the suite behave identically on a
    laptop and in CI. It is deliberately not claimed as a leak control: the fakes
    never open a socket, and every assertion here compares against the fake
    sentinels above."""
    for var in _AZURE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_openai_module(monkeypatch):
    # Reset per test, not at import: state must not bleed between tests.
    _CALLS.clear()
    FakeAzureOpenAI.instances = []
    FakeAsyncAzureOpenAI.instances = []
    FakeAzureOpenAI.raise_exc_on_init = None
    FakeAsyncAzureOpenAI.raise_exc_on_init = None

    module = types.ModuleType("openai")
    module.AzureOpenAI = FakeAzureOpenAI
    module.AsyncAzureOpenAI = FakeAsyncAzureOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return module


def _neutralize_retry_sleep(monkeypatch, *method_names):
    """Patch the retry controller belonging to each named method.

    Every ``@retry``-decorated method owns its own controller, so patching one does
    not cover the others. Async controllers await their ``sleep``, hence the
    coroutine variant for ``_a*`` names. Six attempts of real exponential backoff
    cost roughly 30s per method, so this must be in place *before* any call that is
    expected to fail -- including a construction whose dimension probe fails.
    """

    async def _async_no_sleep(_):
        return None

    for name in method_names:
        controller = getattr(AzureOpenAITextVectorizer, name).retry
        sleep = _async_no_sleep if name.startswith("_a") else (lambda _: None)
        monkeypatch.setattr(controller, "sleep", sleep)


async def _call(vectorizer, method_name, *args, **kwargs):
    """Invoke a sync or async embed entry point uniformly."""
    result = getattr(vectorizer, method_name)(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


# --------------------------------------------------------------------------- #
# Configuration resolution                                                    #
# --------------------------------------------------------------------------- #


def test_init_from_api_config_configures_both_clients(fake_openai_module):
    vectorizer = AzureOpenAITextVectorizer(api_config=_creds())

    client = FakeAzureOpenAI.instances[0]
    aclient = FakeAsyncAzureOpenAI.instances[0]

    assert vectorizer.model == "text-embedding-ada-002"
    assert vectorizer.dtype == "float32"
    assert vectorizer.dims == 3
    assert vectorizer.type == "azure_openai"
    # The two client constructions are copy-pasted in the source; comparing them
    # is the only thing that catches a typo in the second one.
    assert client.kwargs["api_key"] == API_KEY
    assert client.kwargs["api_version"] == API_VERSION
    assert client.kwargs["azure_endpoint"] == ENDPOINT
    assert aclient.kwargs == client.kwargs


def test_init_from_env_vars_configures_both_clients(fake_openai_module, monkeypatch):
    # The env path is what CI actually used before the live resource went away.
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("OPENAI_API_VERSION", API_VERSION)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", API_KEY)

    vectorizer = AzureOpenAITextVectorizer()

    client = FakeAzureOpenAI.instances[0]
    aclient = FakeAsyncAzureOpenAI.instances[0]
    assert vectorizer.dims == 3
    assert client.kwargs["api_key"] == API_KEY
    assert client.kwargs["api_version"] == API_VERSION
    assert client.kwargs["azure_endpoint"] == ENDPOINT
    assert aclient.kwargs == client.kwargs


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, "API endpoint is required"),
        ({"AZURE_OPENAI_ENDPOINT": ENDPOINT}, "API version is required"),
        (
            {
                "AZURE_OPENAI_ENDPOINT": ENDPOINT,
                "OPENAI_API_VERSION": API_VERSION,
            },
            "API key is required",
        ),
    ],
)
def test_missing_config_raises(fake_openai_module, monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=expected):
        AzureOpenAITextVectorizer()

    assert FakeAzureOpenAI.instances == []


def test_extra_api_config_and_kwargs_forwarded_to_clients(fake_openai_module):
    api_config = _creds()
    api_config["organization"] = "org-fake"

    AzureOpenAITextVectorizer(api_config=api_config, timeout=30, max_retries=1)

    client = FakeAzureOpenAI.instances[0]
    aclient = FakeAsyncAzureOpenAI.instances[0]
    assert client.kwargs["organization"] == "org-fake"
    assert client.kwargs["timeout"] == 30
    assert client.kwargs["max_retries"] == 1
    assert aclient.kwargs == client.kwargs


def test_missing_openai_dependency_raises_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pip install openai>=1\.13\.0"):
        AzureOpenAITextVectorizer(api_config=_creds())


# --------------------------------------------------------------------------- #
# Embedding                                                                   #
# --------------------------------------------------------------------------- #


def test_embed_wraps_input_and_forwards_deployment_name(fake_openai_module):
    # Azure's `model` argument is the *deployment* name. A non-default value here
    # is what catches a hardcoded model name in the source.
    vectorizer = AzureOpenAITextVectorizer(
        model="my-embedding-deployment", api_config=_creds()
    )

    result = vectorizer.embed("hello world", user="tester")

    assert result == _vec("hello world")
    assert _CALLS[-1]["input"] == ["hello world"]
    assert _CALLS[-1]["model"] == "my-embedding-deployment"
    assert _CALLS[-1]["kwargs"] == {"user": "tester"}
    assert _CALLS[-1]["async"] is False


def test_embed_many_batches_and_preserves_order(fake_openai_module):
    vectorizer = AzureOpenAITextVectorizer(api_config=_creds())

    result = vectorizer.embed_many(["a", "bb", "ccc", "dddd"], batch_size=2)

    assert result == [_vec("a"), _vec("bb"), _vec("ccc"), _vec("dddd")]
    # _CALLS[0] is __init__'s dimension probe; the rest are the batches.
    assert _CALLS[0]["input"] == ["dimension check"]
    assert [c["input"] for c in _CALLS[1:]] == [["a", "bb"], ["ccc", "dddd"]]


def test_embed_many_empty_makes_no_api_call(fake_openai_module):
    vectorizer = AzureOpenAITextVectorizer(api_config=_creds())
    calls_after_init = len(_CALLS)

    assert vectorizer.embed_many([]) == []
    assert len(_CALLS) == calls_after_init


@pytest.mark.asyncio
async def test_aembed_uses_async_client(fake_openai_module):
    vectorizer = AzureOpenAITextVectorizer(api_config=_creds())

    result = await vectorizer.aembed("hello async")

    assert result == _vec("hello async")
    # Reaching for _client where _aclient belongs is the top copy-paste hazard.
    assert _CALLS[-1]["async"] is True
    assert _CALLS[-1]["input"] == ["hello async"]


@pytest.mark.asyncio
async def test_aembed_many_uses_async_client(fake_openai_module):
    vectorizer = AzureOpenAITextVectorizer(api_config=_creds())

    result = await vectorizer.aembed_many(["a", "bb", "ccc"], batch_size=2)

    assert result == [_vec("a"), _vec("bb"), _vec("ccc")]
    assert [c["input"] for c in _CALLS[1:]] == [["a", "bb"], ["ccc"]]
    assert all(c["async"] is True for c in _CALLS[1:])


# --------------------------------------------------------------------------- #
# Validation and retry                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name,bad_input",
    [
        ("embed", 42),
        ("embed_many", "not a list"),
        ("embed_many", [42]),
        ("aembed", 42),
        ("aembed_many", "not a list"),
        ("aembed_many", [42]),
    ],
)
async def test_type_error_guard_is_not_retried(
    fake_openai_module, method_name, bad_input
):
    vectorizer = AzureOpenAITextVectorizer(api_config=_creds())
    calls_before = len(_CALLS)

    with pytest.raises(TypeError):
        await _call(vectorizer, method_name, bad_input)

    # An unchanged call log proves the guard sits before the `try` block -- no API
    # call is attempted. It does NOT cover retry_if_not_exception_type(TypeError):
    # mutation testing showed that deleting that predicate leaves every test in
    # this file passing, because the TypeError is simply retried six times against
    # a call that never happens. The only observable difference is runtime, ~5s to
    # ~121s of real backoff, and asserting on wall clock would be flaky. So the
    # predicate is knowingly uncovered.
    assert len(_CALLS) == calls_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name,inner_name,args,fake_cls",
    [
        ("embed", "_embed", ("hello",), FakeAzureOpenAI),
        ("embed_many", "_embed_many", (["hello"],), FakeAzureOpenAI),
        ("aembed", "_aembed", ("hello",), FakeAsyncAzureOpenAI),
        ("aembed_many", "_aembed_many", (["hello"],), FakeAsyncAzureOpenAI),
    ],
)
async def test_retry_exhaustion_surfaces_underlying_error(
    fake_openai_module, monkeypatch, method_name, inner_name, args, fake_cls
):
    vectorizer = AzureOpenAITextVectorizer(api_config=_creds())
    _neutralize_retry_sleep(monkeypatch, inner_name)
    # raise_exc is snapshotted at construction, so arm the instance, not the class.
    fake_cls.instances[-1].raise_exc = RuntimeError("503 service unavailable")
    calls_before = len(_CALLS)

    # reraise=True is what makes this a ValueError at all: without it tenacity
    # raises RetryError, which is not a ValueError and would fail this `raises`.
    # The interpolated provider text is part of the contract -- callers log it.
    with pytest.raises(
        ValueError, match=r"Embedding texts? failed: 503 service unavailable"
    ):
        await _call(vectorizer, method_name, *args)

    assert len(_CALLS) - calls_before == 6


def test_probe_failure_during_init_raises_value_error(fake_openai_module, monkeypatch):
    # Patch sleep BEFORE constructing: the probe calls _embed, so an unpatched
    # failing construction burns ~30s of real backoff.
    _neutralize_retry_sleep(monkeypatch, "_embed")
    monkeypatch.setattr(
        FakeAzureOpenAI, "raise_exc_on_init", RuntimeError("deployment not found")
    )

    # Type only, never the message: PR #680 rewrites that string and will own its
    # assertions in `tests/unit/test_vectorizer_dim_errors.py` (added by PR #680,
    # which is still open -- that file does not exist on this branch yet).
    with pytest.raises(ValueError):
        AzureOpenAITextVectorizer(api_config=_creds())

    assert len(_CALLS) == 6


def test_invalid_dtype_uses_base_validation(fake_openai_module):
    with pytest.raises(ValueError, match="Invalid data type"):
        AzureOpenAITextVectorizer(api_config=_creds(), dtype="float25")

    # No clients constructed means no API call happens on a bad dtype.
    assert FakeAzureOpenAI.instances == []


@pytest.mark.parametrize(
    "dtype,itemsize",
    [
        ("float16", 2),
        ("float32", 4),
        ("float64", 8),
        ("bfloat16", 2),
        ("int8", 1),
        ("uint8", 1),
    ],
)
def test_dtype_round_trip(fake_openai_module, dtype, itemsize):
    vectorizer = AzureOpenAITextVectorizer(api_config=_creds(), dtype=dtype)

    assert vectorizer.dtype == dtype
    assert vectorizer.embed("abc") == _vec("abc")

    buffer = vectorizer.embed("abc", as_buffer=True)
    assert isinstance(buffer, bytes)
    assert len(buffer) == 3 * itemsize


# --------------------------------------------------------------------------- #
# api_config handling                                                         #
# --------------------------------------------------------------------------- #


def test_api_config_not_mutated(fake_openai_module):
    # _initialize_clients pops credentials out of api_config, so it must pop from a
    # copy. Same contract, same name as test_google_genai_vectorizer.py.
    api_config = _creds()

    AzureOpenAITextVectorizer(api_config=api_config)

    assert api_config == _creds()
    # Reusable: a second vectorizer from the same dict must still work.
    AzureOpenAITextVectorizer(api_config=api_config)
    assert len(FakeAzureOpenAI.instances) == 2


def test_partial_api_config_falls_back_to_env_per_key(fake_openai_module, monkeypatch):
    """The class docstring promises each credential may come from api_config *or*
    its env var. Resolution is therefore per key, not all-or-nothing."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("OPENAI_API_VERSION", API_VERSION)

    vectorizer = AzureOpenAITextVectorizer(api_config={"api_key": API_KEY})

    assert vectorizer.dims == 3
    client = FakeAzureOpenAI.instances[0]
    assert client.kwargs["api_key"] == API_KEY
    assert client.kwargs["azure_endpoint"] == ENDPOINT
    assert client.kwargs["api_version"] == API_VERSION


# --------------------------------------------------------------------------- #
# Wiring                                                                      #
# --------------------------------------------------------------------------- #


def test_vectorizer_from_dict_supports_azure_openai(fake_openai_module, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("OPENAI_API_VERSION", API_VERSION)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", API_KEY)
    from redisvl.utils.vectorize import vectorizer_from_dict

    vectorizer = vectorizer_from_dict(
        {
            "type": "azure_openai",
            "model": "my-embedding-deployment",
            "dtype": "float64",
        }
    )

    assert isinstance(vectorizer, AzureOpenAITextVectorizer)
    assert vectorizer.model == "my-embedding-deployment"
    assert vectorizer.dtype == "float64"


def test_enum_and_public_export():
    from redisvl.utils.vectorize.base import Vectorizers

    assert Vectorizers("azure_openai") == Vectorizers.azure_openai

    import redisvl.utils.vectorize as vectorize

    assert vectorize.AzureOpenAITextVectorizer is AzureOpenAITextVectorizer


def test_uses_base_public_batch_embedding_methods():
    assert AzureOpenAITextVectorizer.embed_many is BaseVectorizer.embed_many
    assert AzureOpenAITextVectorizer.aembed_many is BaseVectorizer.aembed_many


# --------------------------------------------------------------------------- #
# Guards on the tests themselves                                              #
# --------------------------------------------------------------------------- #


def test_retry_decorated_methods_are_exactly_the_covered_four():
    """Guard for the retry parametrizations above.

    Each ``@retry`` method owns its own controller, and only a test that patches
    that controller's ``sleep`` avoids real exponential backoff. A fifth retried
    method whose test forgot the patch would not fail -- it would make the suite
    hang for minutes. Fail here instead, at the moment the method is added.
    """
    retried = {
        name
        for name in dir(AzureOpenAITextVectorizer)
        if hasattr(getattr(AzureOpenAITextVectorizer, name, None), "retry")
    }
    assert retried == {"_embed", "_embed_many", "_aembed", "_aembed_many"}


def test_fake_matches_real_openai_sdk_contract():
    """Pin the fakes above to the real SDK surface they impersonate.

    Everything else in this file talks to ``FakeAzureOpenAI``, so a breaking change
    in ``openai`` would leave the whole file green while production broke. These are
    exactly the three shapes the source depends on: the ``input``/``model`` keyword
    arguments to ``embeddings.create``, ``response.data``, and ``item.embedding``.
    Skipped locally when ``openai`` is absent; CI installs it via
    ``uv sync --all-extras``.
    """
    pytest.importorskip("openai")

    from openai.resources.embeddings import Embeddings
    from openai.types import CreateEmbeddingResponse, Embedding

    params = inspect.signature(Embeddings.create).parameters
    assert {"input", "model"} <= set(params)
    assert "data" in CreateEmbeddingResponse.model_fields
    assert "embedding" in Embedding.model_fields
