from unittest.mock import MagicMock, patch

import pytest
from redis.exceptions import ConnectionError

from redisvl.redis.connection import RedisConnectionFactory


@pytest.mark.parametrize("use_async", [False, True])
def test_sentinel_url_connection(use_async):
    sentinel_url = (
        "redis+sentinel://username:password@host1:26379,host2:26380/mymaster/0"
    )

    with patch("redisvl.redis.connection.Sentinel") as mock_sentinel:
        mock_master = MagicMock()
        mock_sentinel.return_value.master_for.return_value = mock_master

        if use_async:
            with pytest.warns(DeprecationWarning):
                client = RedisConnectionFactory.get_async_redis_connection(sentinel_url)
        else:
            client = RedisConnectionFactory.get_redis_connection(sentinel_url)

        mock_sentinel.assert_called_once()
        call_args = mock_sentinel.call_args
        assert call_args[0][0] == [("host1", 26379), ("host2", 26380)]
        assert call_args[1]["sentinel_kwargs"] == {
            "username": "username",
            "password": "password",
        }

        mock_sentinel.return_value.master_for.assert_called_once()
        master_for_args = mock_sentinel.return_value.master_for.call_args
        assert master_for_args[0][0] == "mymaster"
        assert master_for_args[1]["db"] == "0"

        assert client == mock_master


@pytest.mark.parametrize("use_async", [False, True])
def test_sentinel_url_connection_no_auth_no_db(use_async):
    sentinel_url = "redis+sentinel://host1:26379,host2:26380/mymaster"

    with patch("redisvl.redis.connection.Sentinel") as mock_sentinel:
        mock_master = MagicMock()
        mock_sentinel.return_value.master_for.return_value = mock_master

        if use_async:
            with pytest.warns(DeprecationWarning):
                client = RedisConnectionFactory.get_async_redis_connection(sentinel_url)
        else:
            client = RedisConnectionFactory.get_redis_connection(sentinel_url)

        mock_sentinel.assert_called_once()
        call_args = mock_sentinel.call_args
        assert call_args[0][0] == [("host1", 26379), ("host2", 26380)]
        assert (
            "sentinel_kwargs" not in call_args[1]
            or call_args[1]["sentinel_kwargs"] == {}
        )

        mock_sentinel.return_value.master_for.assert_called_once()
        master_for_args = mock_sentinel.return_value.master_for.call_args
        assert master_for_args[0][0] == "mymaster"
        assert "db" not in master_for_args[1]

        assert client == mock_master


@pytest.mark.parametrize("use_async", [False, True])
def test_sentinel_url_connection_error(use_async):
    sentinel_url = "redis+sentinel://host1:26379,host2:26380/mymaster"

    with patch("redisvl.redis.connection.Sentinel") as mock_sentinel:
        mock_sentinel.return_value.master_for.side_effect = ConnectionError(
            "Test connection error"
        )

        with pytest.raises(ConnectionError):
            if use_async:
                with pytest.warns(DeprecationWarning):
                    RedisConnectionFactory.get_async_redis_connection(sentinel_url)
            else:
                RedisConnectionFactory.get_redis_connection(sentinel_url)

        mock_sentinel.assert_called_once()
