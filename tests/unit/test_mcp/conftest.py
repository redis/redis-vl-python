import pytest

from redisvl.schema import IndexSchema


@pytest.fixture(scope="session", autouse=True)
def redis_container():
    # Shadow the repo-wide autouse Redis container fixture so MCP unit tests stay
    # pure-unit and do not require Docker; Redis coverage lives in integration tests.
    yield None


def _schema() -> IndexSchema:
    """The one index shape the filter, search, and profile unit tests all build against.

    A plain function rather than a fixture: it is a stateless data builder that
    module-level parametrization and helper functions both need to call, and the
    convention in these files is per-module fakes rather than shared fixtures.
    """
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "docs-index",
                "prefix": "doc",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "category", "type": "tag"},
                {"name": "rating", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )
