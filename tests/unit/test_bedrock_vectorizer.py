import sys
from unittest.mock import MagicMock, patch

import pytest

from redisvl.utils.vectorize.bedrock import BedrockVectorizer


@pytest.fixture
def boto3_mock(monkeypatch):
    for var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
    ):
        monkeypatch.delenv(var, raising=False)
    boto3 = MagicMock()
    with patch.dict(sys.modules, {"boto3": boto3}):
        yield boto3


def _make(**kwargs):
    with patch.object(BedrockVectorizer, "_set_model_dims", return_value=3):
        return BedrockVectorizer(**kwargs)


def test_session_token_is_forwarded(boto3_mock):
    _make(
        api_config={
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
        }
    )
    kwargs = boto3_mock.client.call_args.kwargs
    assert kwargs["aws_session_token"] == "token"
    assert kwargs["aws_access_key_id"] == "AKIA"


def test_env_credentials_left_to_boto3(boto3_mock, monkeypatch):
    # Environment credentials (incl. the session token) are resolved by boto3
    # itself so botocore keeps its refresh support; we must not copy them.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "token")
    _make()
    kwargs = boto3_mock.client.call_args.kwargs
    assert "aws_access_key_id" not in kwargs
    assert "aws_secret_access_key" not in kwargs
    assert "aws_session_token" not in kwargs


def test_default_credential_chain_when_no_keys(boto3_mock):
    _make()
    kwargs = boto3_mock.client.call_args.kwargs
    assert "aws_access_key_id" not in kwargs
    assert "aws_secret_access_key" not in kwargs
    assert "aws_session_token" not in kwargs
    assert kwargs["region_name"] == "us-east-1"


def test_partial_static_keys_raise(boto3_mock):
    with pytest.raises(ValueError, match="together"):
        _make(api_config={"aws_access_key_id": "AKIA"})


def test_env_session_token_not_mixed_with_explicit_keys(boto3_mock, monkeypatch):
    monkeypatch.setenv("AWS_SESSION_TOKEN", "leftover")
    _make(
        api_config={
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "secret",
        }
    )
    kwargs = boto3_mock.client.call_args.kwargs
    assert kwargs["aws_access_key_id"] == "AKIA"
    assert "aws_session_token" not in kwargs


def test_session_token_without_keys_raises(boto3_mock):
    with pytest.raises(ValueError, match="aws_session_token requires"):
        _make(api_config={"aws_session_token": "token"})
    boto3_mock.client.assert_not_called()


def test_conflicting_credentials_raise(boto3_mock):
    with pytest.raises(ValueError, match="Conflicting"):
        _make(
            api_config={
                "aws_access_key_id": "AKIA",
                "aws_secret_access_key": "secret",
            },
            aws_access_key_id="OTHER",
        )
    boto3_mock.client.assert_not_called()


def test_unrelated_boto3_kwargs_forwarded(boto3_mock):
    _make(
        api_config={
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "secret",
        },
        endpoint_url="http://localhost:4566",
    )
    kwargs = boto3_mock.client.call_args.kwargs
    assert kwargs["endpoint_url"] == "http://localhost:4566"
    assert kwargs["aws_access_key_id"] == "AKIA"


def test_no_credentials_error_not_retried(boto3_mock):
    NoCredentialsError = pytest.importorskip("botocore.exceptions").NoCredentialsError

    vec = _make()
    vec._client.invoke_model.side_effect = NoCredentialsError()
    with pytest.raises(NoCredentialsError):
        vec._embed("hello")
    assert vec._client.invoke_model.call_count == 1
    with pytest.raises(NoCredentialsError):
        vec._embed_many(["hello"])
    assert vec._client.invoke_model.call_count == 2


def test_missing_credentials_surface_from_constructor(boto3_mock):
    NoCredentialsError = pytest.importorskip("botocore.exceptions").NoCredentialsError

    boto3_mock.client.return_value.invoke_model.side_effect = NoCredentialsError()
    with pytest.raises(NoCredentialsError):
        BedrockVectorizer()
    assert boto3_mock.client.return_value.invoke_model.call_count == 1


def test_missing_sso_login_not_retried(boto3_mock):
    SSOTokenLoadError = pytest.importorskip("botocore.exceptions").SSOTokenLoadError

    vec = _make()
    vec._client.invoke_model.side_effect = SSOTokenLoadError(error_msg="no token")
    with pytest.raises(SSOTokenLoadError):
        vec._embed("hello")
    assert vec._client.invoke_model.call_count == 1
    with pytest.raises(SSOTokenLoadError):
        vec._embed_many(["hello"])
    assert vec._client.invoke_model.call_count == 2


def test_missing_sso_login_surfaces_from_constructor(boto3_mock):
    SSOTokenLoadError = pytest.importorskip("botocore.exceptions").SSOTokenLoadError

    invoke_model = boto3_mock.client.return_value.invoke_model
    invoke_model.side_effect = SSOTokenLoadError(error_msg="no token")
    with pytest.raises(SSOTokenLoadError):
        BedrockVectorizer()
    assert invoke_model.call_count == 1
