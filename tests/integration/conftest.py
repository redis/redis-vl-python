"""
Conftest for integration tests - uses local Redis Stack instead of Docker.

This allows running integration tests in environments where Docker is not
available but Redis Stack is (e.g., dev containers with redis-stack).
"""

import os

import pytest


# Only override Docker fixtures if Docker is not available
def _docker_available():
    """Check if Docker is available."""
    import shutil
    return shutil.which("docker") is not None


if not _docker_available():
    @pytest.fixture(scope="session", autouse=True)
    def redis_container():
        """
        No-op fixture that overrides the session-scoped redis_container fixture
        when Docker is not available. Uses local Redis Stack instead.
        """
        yield None

    @pytest.fixture(scope="session")
    def redis_url():
        """
        Use local Redis instance (assumes Redis Stack is running on localhost:6379).
        Can be overridden with REDIS_URL environment variable.
        """
        return os.environ.get("REDIS_URL", "redis://localhost:6379")
