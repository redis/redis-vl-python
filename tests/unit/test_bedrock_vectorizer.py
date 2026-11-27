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


@pytest.mark.parametrize("source", ["api_config", "kwargs"])
def test_empty_explicit_keys_raise(boto3_mock, monkeypatch, source):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    empty = {"aws_access_key_id": "", "aws_secret_access_key": ""}
    with pytest.raises(ValueError, match="is empty"):
        _make(**({"api_config": empty} if source == "api_config" else empty))
    boto3_mock.client.assert_not_called()

    # Omitting the keys still leaves resolution to the default chain.
    _make()
    kwargs = boto3_mock.client.call_args.kwargs
    assert "aws_access_key_id" not in kwargs
    assert "aws_secret_access_key" not in kwargs


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


def test_partial_credentials_on_refresh_not_retried(boto3_mock):
    # botocore raises this from invoke_model when refreshing expiring env
    # credentials and one of the variables has gone missing.
    PartialCredentialsError = pytest.importorskip(
        "botocore.exceptions"
    ).PartialCredentialsError

    vec = _make()
    vec._client.invoke_model.side_effect = PartialCredentialsError(
        provider="env", cred_var="AWS_SECRET_ACCESS_KEY"
    )
    with pytest.raises(PartialCredentialsError):
        vec._embed("hello")
    assert vec._client.invoke_model.call_count == 1
    with pytest.raises(PartialCredentialsError):
        vec._embed_many(["hello"])
    assert vec._client.invoke_model.call_count == 2


def test_partial_credentials_surface_from_constructor(boto3_mock):
    PartialCredentialsError = pytest.importorskip(
        "botocore.exceptions"
    ).PartialCredentialsError

    invoke_model = boto3_mock.client.return_value.invoke_model
    invoke_model.side_effect = PartialCredentialsError(
        provider="env", cred_var="AWS_SECRET_ACCESS_KEY"
    )
    with pytest.raises(PartialCredentialsError):
        BedrockVectorizer()
    assert invoke_model.call_count == 1


@pytest.mark.parametrize(
    ("session_profile", "error_name"),
    [(True, "TokenRetrievalError"), (False, "UnauthorizedSSOTokenError")],
    ids=["sso-session", "legacy-sso"],
)
def test_expired_sso_login_surfaces_from_constructor(
    tmp_path, monkeypatch, session_profile, error_name
):
    boto3 = pytest.importorskip("boto3")
    exceptions = pytest.importorskip("botocore.exceptions")
    from botocore.credentials import SSOProvider
    from botocore.stub import Stubber
    from botocore.tokens import SSOTokenProvider
    from botocore.utils import JSONFileCache, SSOTokenLoader
    from tenacity import wait_none

    start_url = "https://example.awsapps.com/start"
    sso_config = f"sso_start_url = {start_url}\nsso_region = us-east-1\n"
    profile = (
        "[profile expired]\n"
        "sso_account_id = 123456789012\n"
        "sso_role_name = TestRole\n"
    )
    if session_profile:
        profile += f"sso_session = expired\n\n[sso-session expired]\n{sso_config}"
    else:
        profile += sso_config

    config_file = tmp_path / "config"
    config_file.write_text(profile)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for provider in (SSOProvider, SSOTokenProvider):
        monkeypatch.setattr(provider, "_SSO_TOKEN_CACHE_DIR", str(tmp_path))
    SSOTokenLoader(JSONFileCache(str(tmp_path))).save_token(
        start_url,
        {"accessToken": "expired", "expiresAt": "2000-01-01T00:00:00Z"},
        session_name="expired" if session_profile else None,
    )

    session = boto3.Session(profile_name="expired")
    bedrock = session.client("bedrock-runtime", region_name="us-east-1")
    sso = session.client("sso", region_name="us-east-1")
    # Older botocore versions let the SSO service reject an expired legacy token.
    stubber = Stubber(sso)
    for _ in range(6):
        stubber.add_client_error("get_role_credentials", "UnauthorizedException")

    with (
        stubber,
        patch.object(session._session, "create_client", return_value=sso),
        patch.object(boto3, "client", return_value=bedrock),
        patch.object(bedrock, "invoke_model", wraps=bedrock.invoke_model) as invoke,
        patch.object(BedrockVectorizer._embed.retry, "wait", wait_none()),
        pytest.raises(getattr(exceptions, error_name)),
    ):
        BedrockVectorizer()
    invoke.assert_called_once()
