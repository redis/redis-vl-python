import pytest


@pytest.fixture(scope="session", autouse=True)
def redis_container():
    # Shadow the repo-wide autouse Redis container fixture so MCP unit tests stay
    # pure-unit and do not require Docker; Redis coverage lives in integration tests.
    yield None
