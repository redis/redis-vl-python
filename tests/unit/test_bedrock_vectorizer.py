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


def test_session_token_from_env(boto3_mock, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "token")
    _make()
    assert boto3_mock.client.call_args.kwargs["aws_session_token"] == "token"


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
