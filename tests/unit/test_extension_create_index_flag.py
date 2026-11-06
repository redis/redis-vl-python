"""
Unit tests for `create_index=False` on the extension constructors.

Every extension calls `create()` while being constructed, which runs `FT.INFO`
first. A credential assembled from `@read`/`@write` is denied `FT.INFO` and
`FT.CREATE` together, so it cannot construct any of them -- even against an
index that already exists and that it can query perfectly well. `create_index`
lets such a caller state what RedisVL is otherwise unable to ask.

The contract is "no index command at all", so these tests assert on the Redis
client rather than on outcomes: the mock records every call, and `ft()` is the
gate every `FT.*` command passes through.
"""

import re
from unittest.mock import MagicMock, Mock

import pytest
from redis import Redis
from redis.exceptions import NoPermissionError

from redisvl.exceptions import RedisSearchError
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.extensions.constants import EXTERNAL_INDEX_DROP_CONFLICT
from redisvl.extensions.message_history import MessageHistory, SemanticMessageHistory
from redisvl.extensions.router import SemanticRouter
from redisvl.extensions.router.schema import Route
from redisvl.index import SearchIndex
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.utils.vectorize import CustomVectorizer

ROUTE = Route(name="greeting", references=["hello"], distance_threshold=0.5)


@pytest.fixture
def vectorizer():
    """A stub vectorizer, so no model is downloaded and no API is called."""
    return CustomVectorizer(lambda text: [0.1, 0.2, 0.3])


def _client() -> MagicMock:
    return MagicMock(spec=Redis)


def _build(kind, client, vectorizer, **kwargs):
    name = kwargs.pop("name", kind)
    if kind == "cache":
        return SemanticCache(
            name=name, vectorizer=vectorizer, redis_client=client, **kwargs
        )
    if kind == "history":
        return MessageHistory(name=name, redis_client=client, **kwargs)
    if kind == "semantic_history":
        return SemanticMessageHistory(
            name=name,
            vectorizer=vectorizer,
            redis_client=client,
            **kwargs,
        )
    return SemanticRouter(
        name=name,
        routes=[ROUTE],
        vectorizer=vectorizer,
        redis_client=client,
        **kwargs,
    )


ALL_KINDS = ["cache", "history", "semantic_history", "router"]
# MessageHistory has no `overwrite` parameter and absorbs one through **kwargs,
# so the contradiction guard cannot reach it.
OVERWRITE_KINDS = ["cache", "semantic_history", "router"]


class TestNoIndexCommandIsIssued:
    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_construction_issues_no_redis_command(self, kind, vectorizer):
        client = _client()
        _build(kind, client, vectorizer, create_index=False)

        client.ft.assert_not_called()
        issued = [call.args[0] for call in client.execute_command.call_args_list]
        assert [cmd for cmd in issued if str(cmd).upper().startswith("FT.")] == []
        # Deliberately stricter than the stated contract: nothing at all should
        # touch Redis during construction, so a new eager call has to be
        # justified here rather than slipping in.
        assert client.mock_calls == []

    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_the_flag_is_kept_as_instance_state(self, kind, vectorizer):
        # Three of the four never read it back, so without this a dead-store
        # check prunes it -- and it is what a future "index not owned" notion
        # would build on.
        built = _build(kind, _client(), vectorizer, create_index=False)
        assert built._create_index is False

    def test_router_writes_no_stored_config(self, vectorizer):
        # The router's config blob is the source of truth for from_existing(),
        # and is written from an unverified local route list. Rewriting it would
        # truncate a shared router's routes.
        client = _client()
        router = _build("router", client, vectorizer, create_index=False)
        client.json.assert_not_called()
        # And the flag itself must never become part of that blob.
        assert "create_index" not in router.to_dict()


class TestLazyConnectionAfterSkippingCreate:
    """`create()` used to be the de-facto eager connect.

    `SearchIndex.client` stays `None` until the lazy `_redis_client` property
    runs, so with the existence check skipped, any extension method reaching for
    the raw client would dereference `None`. These pin the two shapes that broke.
    """

    def test_history_drop_connects_lazily(self, vectorizer, monkeypatch):
        client = _client()
        monkeypatch.setattr(
            RedisConnectionFactory, "get_redis_connection", lambda **kwargs: client
        )
        history = MessageHistory(name="history", create_index=False)
        assert history._index.client is None  # the precondition that broke

        history.drop(id="abc")

        client.delete.assert_called_once()

    def test_router_reference_lookup_connects_lazily(self, vectorizer, monkeypatch):
        client = _client()
        client.scan.return_value = (0, [])
        monkeypatch.setattr(
            RedisConnectionFactory, "get_redis_connection", lambda **kwargs: client
        )
        router = SemanticRouter(
            name="router", routes=[ROUTE], vectorizer=vectorizer, create_index=False
        )
        assert router._index.client is None

        assert router.get_route_references(route_name="greeting") == []


class TestContradictoryArguments:
    @pytest.mark.parametrize("kind", OVERWRITE_KINDS)
    def test_overwrite_with_create_index_false_is_rejected(self, kind, vectorizer):
        with pytest.raises(ValueError, match="contradict"):
            _build(kind, _client(), vectorizer, create_index=False, overwrite=True)


class TestExternalIndexLifecycle:
    """The flag guards the index's lifecycle, not its contents.

    `delete()` drops the index, so an attach-only instance must refuse it.
    `clear()` removes entries and leaves the index standing, so it does not.
    """

    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_dropping_the_index_is_rejected(self, kind, vectorizer):
        client = _client()
        extension = _build(
            kind,
            client,
            vectorizer,
            create_index=False,
            name="production_alias",
        )

        with pytest.raises(ValueError, match=re.escape(EXTERNAL_INDEX_DROP_CONFLICT)):
            extension.delete()

        assert client.mock_calls == []

    @pytest.mark.asyncio
    async def test_async_cache_dropping_the_index_is_rejected(self, vectorizer):
        client = _client()
        cache = SemanticCache(
            name="production_alias",
            vectorizer=vectorizer,
            redis_client=client,
            create_index=False,
        )

        with pytest.raises(ValueError, match=re.escape(EXTERNAL_INDEX_DROP_CONFLICT)):
            await cache.adelete()

        assert client.mock_calls == []

    def test_cache_clear_issues_no_index_command(self, vectorizer):
        # Asserted as "ft() was never reached" rather than on the SCAN call
        # shape, which belongs to BaseCache and is being reworked separately.
        client = _client()
        client.scan.return_value = (0, ["llmcache:abc"])
        client.scan_iter.return_value = iter(["llmcache:abc"])
        cache = SemanticCache(
            name="llmcache",
            vectorizer=vectorizer,
            redis_client=client,
            create_index=False,
        )

        cache.clear()

        client.ft.assert_not_called()
        client.delete.assert_called_once_with("llmcache:abc")

    @pytest.mark.parametrize("kind", ["history", "semantic_history", "router"])
    def test_index_backed_clear_is_not_refused(self, kind, vectorizer, monkeypatch):
        # These delegate to SearchIndex.clear(), so the minimal assertion is
        # that control reaches it at all.
        cleared = Mock(return_value=0)
        monkeypatch.setattr(SearchIndex, "clear", cleared)
        extension = _build(
            kind,
            _client(),
            vectorizer,
            create_index=False,
            name="production_alias",
        )

        extension.clear()

        cleared.assert_called_once()


class TestRouterWithoutRoutes:
    def test_empty_routes_raises_a_useful_error_when_matching(self, vectorizer):
        # The guard is unconditional -- `routes=[]` is legal on either path, and
        # the flag is about index ownership, not about route contents. Only the
        # create_index=False case can be hermetic; the default path needs a live
        # server and is covered by the integration suite.
        router = SemanticRouter(
            name="router",
            routes=[],
            vectorizer=vectorizer,
            redis_client=_client(),
            create_index=False,
        )
        with pytest.raises(ValueError, match="no routes"):
            router("hello")


class TestFromExisting:
    @pytest.mark.parametrize("with_client", [False, True])
    def test_from_existing_rejects_create_index_false_with_overwrite(self, with_client):
        kwargs = {"redis_client": _client()} if with_client else {}
        with pytest.raises(ValueError, match="contradict"):
            SemanticRouter.from_existing(
                "router", create_index=False, overwrite=True, **kwargs
            )

    def test_from_existing_issues_no_index_command(self, vectorizer, monkeypatch):
        # SemanticRouter.from_existing reads the stored config with JSON.GET, so
        # with the flag threaded it needs no FT.INFO -- which makes it the way to
        # attach to a router under a credential that cannot run one.
        stored = _build("router", _client(), vectorizer, create_index=False).to_dict()
        monkeypatch.setattr(
            "redisvl.utils.vectorize.vectorizer_from_dict", lambda _: vectorizer
        )

        # A real instance, so validate_sync_redis's issubclass check passes.
        client = Redis.from_url("redis://localhost:6379")
        client.client_setinfo = Mock()
        client.ft = Mock()
        client.json = Mock()
        client.json.return_value.get.return_value = stored

        router = SemanticRouter.from_existing(
            "router", redis_client=client, create_index=False
        )

        assert [route.name for route in router.routes] == ["greeting"]
        client.ft.assert_not_called()

    def test_from_existing_still_verifies_by_default(self, vectorizer, monkeypatch):
        # A stored route config proves a config key exists, not that the index
        # does -- so the default must still check.
        stored = _build("router", _client(), vectorizer, create_index=False).to_dict()
        monkeypatch.setattr(
            "redisvl.utils.vectorize.vectorizer_from_dict", lambda _: vectorizer
        )

        client = Redis.from_url("redis://localhost:6379")
        client.client_setinfo = Mock()
        client.json = Mock()
        client.json.return_value.get.return_value = stored
        # Stand in for the credential this feature exists for: the default path
        # must reach FT.INFO, so a denial has to surface.
        client.ft = Mock()
        client.ft.return_value.info.side_effect = NoPermissionError(
            "User acl_user has no permissions to run the 'FT.INFO' command"
        )

        with pytest.raises(RedisSearchError):
            SemanticRouter.from_existing("router", redis_client=client)

        client.ft.assert_called()
