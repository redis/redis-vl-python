"""Errors from vectorizer ``_set_model_dims()`` become an actionable ``ValueError``.

Each vectorizer probes its provider with a throwaway embedding call to learn the
model's dimensionality. When that probe fails, ``_set_model_dims()`` wraps
whatever it catches in a ``ValueError`` that names the provider and the model,
chained with ``from e`` so the original exception (SDK error, retry exhaustion,
etc.) stays visible in the traceback rather than being swallowed.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest


def _openai_error(cls, status):
    """Build an openai SDK error without performing a request."""
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


# ---------------------------------------------------------------------------
# End-to-end: the real _embed()/_initialize_client() code runs.
# ---------------------------------------------------------------------------


def test_openai_dim_probe_names_the_model_and_chains_the_cause(monkeypatch):
    """OpenAI's real client.embeddings.create() raises, _embed() wraps it, and
    _set_model_dims() must still report the model and preserve the cause."""
    import time

    import openai

    from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

    # _embed() is @retry-decorated and does not exempt ValueError from retrying,
    # so a permanent failure like a 401 is retried up to 6 times with exponential
    # backoff before RetryError is raised. Skip the real sleeps -- the retrying
    # itself isn't what this test is checking.
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

    error = _openai_error(openai.AuthenticationError, 401)
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = error

    with patch.object(
        OpenAITextVectorizer,
        "_initialize_clients",
        lambda self, *a, **k: setattr(self, "_client", mock_client),
    ):
        with pytest.raises(ValueError) as excinfo:
            OpenAITextVectorizer(model="text-embedding-3-small")

    message = str(excinfo.value)
    assert "text-embedding-3-small" in message
    assert excinfo.value.__cause__ is not None


def test_bedrock_dim_probe_names_the_model_and_chains_the_cause(monkeypatch):
    """Bedrock's real client.invoke_model() raises a ClientError, _embed() wraps
    it, and _set_model_dims() must still report the model and preserve the cause."""
    import time

    from botocore.exceptions import ClientError

    from redisvl.utils.vectorize.bedrock import BedrockVectorizer

    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

    denied = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "nope"}}, "InvokeModel"
    )
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = denied

    with patch.object(
        BedrockVectorizer,
        "_initialize_client",
        lambda self, *a, **k: setattr(self, "_client", mock_client),
    ):
        with pytest.raises(ValueError) as excinfo:
            BedrockVectorizer(model="amazon.titan-embed-text-v2:0")

    message = str(excinfo.value)
    assert "amazon.titan-embed-text-v2:0" in message
    assert excinfo.value.__cause__ is not None


def test_huggingface_dim_probe_reports_local_model_load_failure():
    """A HuggingFace model that fails to load raises OSError inside the real
    SentenceTransformer() construction, in _initialize_client() -- before
    _set_model_dims() ever runs. This must be caught where it actually happens."""
    from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

    with patch(
        "sentence_transformers.SentenceTransformer",
        side_effect=OSError("no such file"),
    ):
        with pytest.raises(ValueError) as excinfo:
            HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")

    message = str(excinfo.value)
    assert "sentence-transformers/all-mpnet-base-v2" in message
    assert "downloaded" in message


# ---------------------------------------------------------------------------
# Wrap-and-chain: _embed() is patched to raise directly, pinning that
# _set_model_dims() names the provider/model and chains the real cause via
# `from e` rather than losing it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vectorizer_path, class_name, init_method, model",
    [
        (
            "redisvl.utils.vectorize.text.azureopenai",
            "AzureOpenAITextVectorizer",
            "_initialize_clients",
            "my-deployment",
        ),
        (
            "redisvl.utils.vectorize.text.cohere",
            "CohereTextVectorizer",
            "_initialize_client",
            "embed-english-v3.0",
        ),
        (
            "redisvl.utils.vectorize.text.mistral",
            "MistralAITextVectorizer",
            "_initialize_client",
            "mistral-embed",
        ),
        (
            "redisvl.utils.vectorize.vertexai",
            "VertexAIVectorizer",
            "_initialize_client",
            "text-embedding-004",
        ),
    ],
)
def test_dim_probe_names_the_model_and_chains_the_cause(
    vectorizer_path, class_name, init_method, model
):
    import importlib

    module = importlib.import_module(vectorizer_path)
    vectorizer_cls = getattr(module, class_name)

    cause = RuntimeError("boom from the SDK")

    with patch.object(vectorizer_cls, init_method, lambda self, *a, **k: None):
        with patch.object(vectorizer_cls, "_embed", side_effect=cause):
            with pytest.raises(ValueError) as excinfo:
                vectorizer_cls(model=model)

    message = str(excinfo.value)
    assert model in message
    assert excinfo.value.__cause__ is cause


def test_voyageai_dim_probe_names_the_model_and_chains_the_cause():
    from redisvl.utils.vectorize.voyageai import VoyageAIVectorizer

    cause = RuntimeError("boom from the SDK")

    def _fake_init(self, *a, **k):
        # _setup() reaches into self._client / self._aclient right after
        # _initialize_client() returns (to grab .embed / .multimodal_embed), so
        # the no-op stub has to leave both set rather than leaving them unset.
        self._client = MagicMock()
        self._aclient = MagicMock()

    with patch.object(VoyageAIVectorizer, "_initialize_client", _fake_init):
        with patch.object(VoyageAIVectorizer, "_embed", side_effect=cause):
            with pytest.raises(ValueError) as excinfo:
                VoyageAIVectorizer(model="voyage-3")

    message = str(excinfo.value)
    assert "voyage-3" in message
    assert excinfo.value.__cause__ is cause


def test_voyageai_dim_probe_catches_bad_model_id_type_error():
    """VoyageAI's _embed_many() re-raises InvalidRequestError as TypeError --
    deliberately, so retry_if_not_exception_type(TypeError) skips retrying it,
    since a bad model id can never succeed no matter how many attempts. This
    drives the real _embed_many() code (only the client's .embed() call is
    stubbed) to prove the TypeError path is still caught by the generic
    except Exception clause."""
    import voyageai.error

    from redisvl.utils.vectorize.voyageai import VoyageAIVectorizer

    bad_model = voyageai.error.InvalidRequestError("model not found")
    mock_client = MagicMock()
    mock_client.embed.side_effect = bad_model

    def _fake_init(self, *a, **k):
        self._client = mock_client
        self._aclient = mock_client

    with patch.object(VoyageAIVectorizer, "_initialize_client", _fake_init):
        with pytest.raises(ValueError) as excinfo:
            VoyageAIVectorizer(model="not-a-real-voyage-model")

    message = str(excinfo.value)
    assert "not-a-real-voyage-model" in message
    assert excinfo.value.__cause__ is not None


def test_unanticipated_errors_still_become_valueerror():
    """The generic fallback must survive: no error may escape raw."""
    from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

    with patch.object(
        OpenAITextVectorizer, "_initialize_clients", lambda self, *a, **k: None
    ):
        with patch.object(
            OpenAITextVectorizer, "_embed", side_effect=ZeroDivisionError("surprise")
        ):
            with pytest.raises(ValueError) as excinfo:
                OpenAITextVectorizer(model="text-embedding-3-small")

    assert "text-embedding-3-small" in str(excinfo.value)
