"""Provider-specific error messages from vectorizer ``_set_model_dims()``.

Each vectorizer probes its provider with a throwaway embedding call to learn the
model's dimensionality. When that probe fails, the resulting ``ValueError`` should
name the provider, the model, and what the caller can do about it -- not just
restate the SDK's own message.

Every vectorizer is driven through its real ``__init__`` with the network client
stubbed out, so these tests exercise the same path a user hits on a bad API key.
"""

from unittest.mock import patch

import httpx
import pytest


def _openai_error(cls, status):
    """Build an openai SDK error without performing a request."""
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize(
    "status, error_name, expected",
    [
        (401, "AuthenticationError", "OPENAI_API_KEY"),
        (404, "NotFoundError", "does not recognize the embedding model"),
    ],
)
def test_openai_dim_probe_reports_provider_and_remediation(
    status, error_name, expected
):
    import openai

    from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

    error = _openai_error(getattr(openai, error_name), status)

    with patch.object(
        OpenAITextVectorizer, "_initialize_clients", lambda self, *a, **k: None
    ):
        with patch.object(OpenAITextVectorizer, "_embed", side_effect=error):
            with pytest.raises(ValueError) as excinfo:
                OpenAITextVectorizer(model="text-embedding-3-small")

    message = str(excinfo.value)
    assert "text-embedding-3-small" in message
    assert expected in message


def test_azure_openai_dim_probe_names_the_deployment():
    import openai

    from redisvl.utils.vectorize.text.azureopenai import AzureOpenAITextVectorizer

    error = _openai_error(openai.NotFoundError, 404)

    with patch.object(
        AzureOpenAITextVectorizer, "_initialize_clients", lambda self, *a, **k: None
    ):
        with patch.object(AzureOpenAITextVectorizer, "_embed", side_effect=error):
            with pytest.raises(ValueError) as excinfo:
                AzureOpenAITextVectorizer(model="my-deployment")

    message = str(excinfo.value)
    assert "my-deployment" in message
    # Azure addresses models by deployment name; the message must say so.
    assert "deployment" in message


def test_bedrock_dim_probe_distinguishes_auth_from_bad_model_id():
    from botocore.exceptions import ClientError

    from redisvl.utils.vectorize.bedrock import BedrockVectorizer

    denied = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "nope"}}, "InvokeModel"
    )

    with patch.object(
        BedrockVectorizer, "_initialize_client", lambda self, *a, **k: None
    ):
        with patch.object(BedrockVectorizer, "_embed", side_effect=denied):
            with pytest.raises(ValueError) as excinfo:
                BedrockVectorizer(model="amazon.titan-embed-text-v2:0")

    message = str(excinfo.value)
    assert "amazon.titan-embed-text-v2:0" in message
    assert "bedrock:InvokeModel" in message


def test_bedrock_dim_probe_reports_unknown_model_id():
    from botocore.exceptions import ClientError

    from redisvl.utils.vectorize.bedrock import BedrockVectorizer

    missing = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}},
        "InvokeModel",
    )

    with patch.object(
        BedrockVectorizer, "_initialize_client", lambda self, *a, **k: None
    ):
        with patch.object(BedrockVectorizer, "_embed", side_effect=missing):
            with pytest.raises(ValueError) as excinfo:
                BedrockVectorizer(model="not-a-real-model")

    message = str(excinfo.value)
    assert "not-a-real-model" in message
    assert "AWS_REGION" in message


def test_huggingface_dim_probe_reports_local_model_load_failure():
    from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

    with patch.object(
        HFTextVectorizer, "_initialize_client", lambda self, *a, **k: None
    ):
        with patch.object(
            HFTextVectorizer, "_embed", side_effect=OSError("no such file")
        ):
            with pytest.raises(ValueError) as excinfo:
                HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")

    message = str(excinfo.value)
    assert "sentence-transformers/all-mpnet-base-v2" in message
    assert "downloaded" in message


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
