"""Provider-specific error messages from vectorizer ``_set_model_dims()``.

Each vectorizer probes its provider with a throwaway embedding call to learn the
model's dimensionality. When that probe fails, the resulting ``ValueError`` should
name the provider, the model, and what the caller can do about it -- not just
restate the SDK's own message.

``_embed()``/``_embed_many()`` on every provider already catch the SDK's own
exception and re-wrap it in a generic ``ValueError`` before ``_set_model_dims()``
ever sees it. That means ``_set_model_dims()`` cannot simply catch the SDK's
exception type directly -- it has to unwrap the ``ValueError`` it receives and
dispatch on ``__cause__``/``__context__`` instead. A test that patches ``_embed``
with ``side_effect=<raw SDK error>`` skips that wrapping entirely and would pass
even if the dispatch logic were broken, since it hands ``_set_model_dims`` an
exception shape production code never produces.

So this file mixes two kinds of test:

- True end-to-end tests (OpenAI, Bedrock, HuggingFace) that patch only the
  network client / model-loading call, so the vectorizer's real ``_embed()`` /
  ``_initialize_client()`` code runs and performs the real wrapping.
- Cause-dispatch tests for the remaining providers, which hand ``_embed`` a
  ``ValueError`` built the same way ``_embed`` really builds one -- raised while
  handling the SDK's exception, so ``__context__`` is set by Python's normal
  implicit exception chaining, not fabricated by the test.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest


def _openai_error(cls, status):
    """Build an openai SDK error without performing a request."""
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


def _wrapped(cause: BaseException, message: str = "wrapped") -> ValueError:
    """Build a ValueError whose __context__ is `cause`.

    This mirrors exactly what `_embed()`/`_embed_many()` produce: they catch the
    SDK's exception and raise a generic ValueError while still handling it, which
    is what makes Python set __context__ via implicit chaining. Constructing the
    ValueError outside of an active `except cause` block would leave __context__
    unset, so `cause` is actually raised and caught here rather than merely
    referenced.
    """
    try:
        raise cause
    except type(cause):
        try:
            raise ValueError(message)
        except ValueError as wrapped_error:
            return wrapped_error


# ---------------------------------------------------------------------------
# End-to-end: the real _embed()/_initialize_client() code runs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, error_name, expected",
    [
        (401, "AuthenticationError", "OPENAI_API_KEY"),
        (404, "NotFoundError", "does not recognize the embedding model"),
    ],
)
def test_openai_dim_probe_reports_provider_and_remediation(
    monkeypatch, status, error_name, expected
):
    """OpenAI's real client.embeddings.create() raises, _embed() wraps it, and
    _set_model_dims() must still recover the real cause and report on it."""
    import time

    import openai

    from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

    # _embed() is @retry-decorated and does not exempt ValueError from retrying,
    # so a permanent failure like a 401 is retried up to 6 times with exponential
    # backoff before RetryError is raised. Skip the real sleeps -- the retrying
    # itself isn't what this test is checking.
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

    error = _openai_error(getattr(openai, error_name), status)
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
    assert expected in message


def test_bedrock_dim_probe_distinguishes_auth_from_bad_model_id(monkeypatch):
    """Bedrock's real client.invoke_model() raises a ClientError, _embed() wraps
    it, and _set_model_dims() must still branch on the AWS error code."""
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
    assert "bedrock:InvokeModel" in message


def test_bedrock_dim_probe_reports_unknown_model_id(monkeypatch):
    import time

    from botocore.exceptions import ClientError

    from redisvl.utils.vectorize.bedrock import BedrockVectorizer

    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

    missing = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}},
        "InvokeModel",
    )
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = missing

    with patch.object(
        BedrockVectorizer,
        "_initialize_client",
        lambda self, *a, **k: setattr(self, "_client", mock_client),
    ):
        with pytest.raises(ValueError) as excinfo:
            BedrockVectorizer(model="not-a-real-model")

    message = str(excinfo.value)
    assert "not-a-real-model" in message
    assert "AWS_REGION" in message


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
# Cause-dispatch: _embed() is patched to raise the same shape of ValueError it
# really raises (built via _wrapped(), not a raw SDK exception), so these pin
# _set_model_dims()'s unwrap-and-dispatch logic in isolation.
# ---------------------------------------------------------------------------


def test_azure_openai_dim_probe_names_the_deployment():
    import openai

    from redisvl.utils.vectorize.text.azureopenai import AzureOpenAITextVectorizer

    wrapped = _wrapped(_openai_error(openai.NotFoundError, 404))

    with patch.object(
        AzureOpenAITextVectorizer, "_initialize_clients", lambda self, *a, **k: None
    ):
        with patch.object(AzureOpenAITextVectorizer, "_embed", side_effect=wrapped):
            with pytest.raises(ValueError) as excinfo:
                AzureOpenAITextVectorizer(model="my-deployment")

    message = str(excinfo.value)
    assert "my-deployment" in message
    # Azure addresses models by deployment name; the message must say so.
    assert "deployment" in message


def test_cohere_dim_probe_reports_unauthorized():
    import cohere

    from redisvl.utils.vectorize.text.cohere import CohereTextVectorizer

    wrapped = _wrapped(cohere.UnauthorizedError("nope"))

    with patch.object(
        CohereTextVectorizer, "_initialize_client", lambda self, *a, **k: None
    ):
        with patch.object(CohereTextVectorizer, "_embed", side_effect=wrapped):
            with pytest.raises(ValueError) as excinfo:
                CohereTextVectorizer(model="embed-english-v3.0")

    message = str(excinfo.value)
    assert "embed-english-v3.0" in message
    assert "COHERE_API_KEY" in message


def test_mistral_dim_probe_reports_sdk_error():
    from mistralai.models import SDKError

    from redisvl.utils.vectorize.text.mistral import MistralAITextVectorizer

    wrapped = _wrapped(
        SDKError(
            "nope",
            raw_response=httpx.Response(
                401,
                request=httpx.Request("POST", "https://api.mistral.ai/v1/embeddings"),
            ),
        )
    )

    with patch.object(
        MistralAITextVectorizer, "_initialize_client", lambda self, *a, **k: None
    ):
        with patch.object(MistralAITextVectorizer, "_embed", side_effect=wrapped):
            with pytest.raises(ValueError) as excinfo:
                MistralAITextVectorizer(model="mistral-embed")

    message = str(excinfo.value)
    assert "mistral-embed" in message
    assert "MISTRAL_API_KEY" in message


def test_vertexai_dim_probe_reports_permission_denied():
    from google.api_core.exceptions import PermissionDenied

    from redisvl.utils.vectorize.vertexai import VertexAIVectorizer

    wrapped = _wrapped(PermissionDenied("nope"))

    with patch.object(
        VertexAIVectorizer, "_initialize_client", lambda self, *a, **k: None
    ):
        with patch.object(VertexAIVectorizer, "_embed", side_effect=wrapped):
            with pytest.raises(ValueError) as excinfo:
                VertexAIVectorizer(model="text-embedding-004")

    message = str(excinfo.value)
    assert "text-embedding-004" in message
    assert "GOOGLE_APPLICATION_CREDENTIALS" in message


def test_voyageai_dim_probe_reports_authentication_error():
    import voyageai.error

    from redisvl.utils.vectorize.voyageai import VoyageAIVectorizer

    wrapped = _wrapped(voyageai.error.AuthenticationError("nope"))

    def _fake_init(self, *a, **k):
        # _setup() reaches into self._client / self._aclient right after
        # _initialize_client() returns (to grab .embed / .multimodal_embed), so
        # the no-op stub has to leave both set rather than leaving them unset.
        self._client = MagicMock()
        self._aclient = MagicMock()

    with patch.object(VoyageAIVectorizer, "_initialize_client", _fake_init):
        with patch.object(VoyageAIVectorizer, "_embed", side_effect=wrapped):
            with pytest.raises(ValueError) as excinfo:
                VoyageAIVectorizer(model="voyage-3")

    message = str(excinfo.value)
    assert "voyage-3" in message
    assert "VOYAGE_API_KEY" in message


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
