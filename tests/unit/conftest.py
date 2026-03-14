"""
Conftest for unit tests - overrides Docker fixtures to allow tests to run
without Docker/Redis infrastructure.

Unit tests should test pure logic without external dependencies.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def redis_container():
    """
    No-op fixture that overrides the session-scoped redis_container fixture
    from the root conftest.py. This allows unit tests to run without Docker.
    """
    yield None


@pytest.fixture(scope="session")
def redis_url():
    """
    Dummy redis_url fixture for unit tests.
    Unit tests should not depend on this - if they do, they should be
    integration tests instead.
    """
    return "redis://localhost:6379"  # Not actually used
