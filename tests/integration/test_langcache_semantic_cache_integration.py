"""Integration tests for LangCacheSemanticCache against the LangCache managed service.

These tests exercise the real LangCache API using two configured caches:
- One with attributes configured
- One without attributes configured

Both caches are shared, and by more writers than is obvious: pytest-xdist
spreads this suite across workers within a run (``make test-all`` uses
``-n auto``), and every CI run reaching the same ``cache_id`` from repo secrets
-- pull requests via ``.github/workflows/test.yml``, forks via
``test-fork-pr.yml``, pushes to main, and the nightly cron -- runs it again
concurrently. So the rules here are:

- No test may flush a whole cache. ``delete()``/``clear()``/``adelete()``/
  ``aclear()`` wipe every entry, including ones another worker or another PR's
  job stored moments earlier. This is a constraint, not a preference: there is
  no safe way to test a global flush against a cache we do not exclusively own,
  so the flush wrappers are covered against a mocked SDK in
  tests/unit/test_langcache_semantic_cache.py, and the flush HTTP path itself
  is deliberately left untested.
- Every test that writes tags its prompts, responses, and attribute values with
  a unique ``scope`` token, so no test can observe or delete another's entries.
- Assert that your own scoped entry is (or is no longer) among the hits -- never
  on ``hits[0]`` and never on ``hits`` being empty. LangCache returns one result
  by default and prompts here differ only by the scope token, so a concurrent
  run's semantically identical prompt is a legitimate candidate for that slot.
  Where a test's subject is not retrieval itself, filter on a scoped attribute
  so the service can only return your own entries.
- Write every entry with a TTL, so the shared caches self-clean without anyone
  flushing them. Note that a TTL set on the constructor is silently ignored by
  ``store()``, so it has to be passed per call -- do not hoist it into the
  fixtures until that is fixed.

Env vars (loaded from .env locally, injected via CI):
- LANGCACHE_WITH_ATTRIBUTES_API_KEY
- LANGCACHE_WITH_ATTRIBUTES_CACHE_ID
- LANGCACHE_WITH_ATTRIBUTES_URL
- LANGCACHE_NO_ATTRIBUTES_API_KEY
- LANGCACHE_NO_ATTRIBUTES_CACHE_ID
- LANGCACHE_NO_ATTRIBUTES_URL
"""

import asyncio
import os
import time
import uuid

import pytest
from dotenv import load_dotenv

from redisvl.extensions.cache.llm.langcache import LangCacheSemanticCache

load_dotenv()

REQUIRED_WITH_ATTRS_VARS = (
    "LANGCACHE_WITH_ATTRIBUTES_API_KEY",
    "LANGCACHE_WITH_ATTRIBUTES_CACHE_ID",
    "LANGCACHE_WITH_ATTRIBUTES_URL",
)

REQUIRED_NO_ATTRS_VARS = (
    "LANGCACHE_NO_ATTRIBUTES_API_KEY",
    "LANGCACHE_NO_ATTRIBUTES_CACHE_ID",
    "LANGCACHE_NO_ATTRIBUTES_URL",
)

# TTL for entries whose own lifetime is not under test, so the shared caches
# drain on their own. Comfortably outlasts the slowest test that reads back what
# it wrote, while keeping concurrent runs' leftovers short-lived.
TEST_ENTRY_TTL = 60


def _require_env_vars(var_names: tuple[str, ...]) -> dict[str, str]:
    missing = [name for name in var_names if not os.getenv(name)]
    if missing:
        pytest.skip(
            f"Missing required LangCache env vars: {', '.join(missing)}. "
            "Set them locally (e.g., via .env) or in CI secrets to run these tests."
        )

    return {name: os.environ[name] for name in var_names}


@pytest.fixture
def scope() -> str:
    """Token unique to each test invocation; see the module docstring for why."""

    return uuid.uuid4().hex[:12]


@pytest.fixture(autouse=True)
def _ban_whole_cache_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the no-flush rule mechanical rather than a convention."""

    for method in ("delete", "adelete", "clear", "aclear"):
        monkeypatch.setattr(
            LangCacheSemanticCache,
            method,
            lambda *args, **kwargs: pytest.fail(
                "Whole-cache flush is banned in this suite -- it wipes other "
                "workers' and other CI runs' entries. See the module docstring."
            ),
        )


@pytest.fixture
def langcache_with_attrs() -> LangCacheSemanticCache:
    """LangCacheSemanticCache instance bound to a cache with attributes configured."""

    env = _require_env_vars(REQUIRED_WITH_ATTRS_VARS)

    return LangCacheSemanticCache(
        name="langcache_with_attributes",
        server_url=env["LANGCACHE_WITH_ATTRIBUTES_URL"],
        cache_id=env["LANGCACHE_WITH_ATTRIBUTES_CACHE_ID"],
        api_key=env["LANGCACHE_WITH_ATTRIBUTES_API_KEY"],
    )


@pytest.fixture
def langcache_no_attrs() -> LangCacheSemanticCache:
    """LangCacheSemanticCache instance bound to a cache with NO attributes configured."""

    env = _require_env_vars(REQUIRED_NO_ATTRS_VARS)

    return LangCacheSemanticCache(
        name="langcache_no_attributes",
        server_url=env["LANGCACHE_NO_ATTRIBUTES_URL"],
        cache_id=env["LANGCACHE_NO_ATTRIBUTES_CACHE_ID"],
        api_key=env["LANGCACHE_NO_ATTRIBUTES_API_KEY"],
    )


@pytest.mark.requires_api_keys
class TestLangCacheSemanticCacheIntegrationWithAttributes:
    def test_store_and_check_sync(
        self, langcache_with_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        prompt = f"What is Redis? [{scope}]"
        response = f"Redis is an in-memory data store. [{scope}]"

        entry_id = langcache_with_attrs.store(
            prompt=prompt, response=response, ttl=TEST_ENTRY_TTL
        )
        assert entry_id

        # Deliberately unfiltered: exact retrieval of a just-stored prompt is
        # what this test is for, and the scoped prompt is what makes the exact
        # strategy (tried before semantic) able to single it out.
        hits = langcache_with_attrs.check(prompt=prompt)
        assert any(
            hit["prompt"] == prompt and hit["response"] == response for hit in hits
        ), f"scoped entry not returned; got {hits}"

    def test_store_with_per_entry_ttl_expires(
        self, langcache_with_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        """Per-entry TTL should cause individual entries to expire."""

        prompt = f"Per-entry TTL test [{scope}]"
        response = f"This entry should expire quickly. [{scope}]"
        # Filtering on a scoped attribute makes the result set provably this
        # test's own, so neither assertion depends on how the service ranks a
        # concurrent run's near-identical prompt.
        metadata = {"user_id": f"tenant_ttl_{scope}"}

        # The TTL has to outlast a store round trip plus a search round trip
        # against a shared managed service, or the entry can expire before the
        # pre-expiry assertion runs.
        entry_id = langcache_with_attrs.store(
            prompt=prompt,
            response=response,
            metadata=metadata,
            ttl=5,
        )
        assert entry_id

        # Immediately after storing, the entry should be retrievable.
        hits = langcache_with_attrs.check(prompt=prompt, attributes=metadata)
        assert any(
            hit["response"] == response for hit in hits
        ), f"entry not retrievable before its TTL elapsed; got {hits}"

        # Wait for TTL to elapse and confirm the entry is no longer returned.
        time.sleep(6)

        hits_after_ttl = langcache_with_attrs.check(
            prompt=prompt, attributes=metadata, num_results=5
        )
        assert not any(hit["response"] == response for hit in hits_after_ttl)

    @pytest.mark.asyncio
    async def test_store_and_check_async(
        self, langcache_with_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        prompt = f"What is Redis async? [{scope}]"
        response = f"Redis is an in-memory data store (async). [{scope}]"

        entry_id = await langcache_with_attrs.astore(
            prompt=prompt, response=response, ttl=TEST_ENTRY_TTL
        )
        assert entry_id

        hits = await langcache_with_attrs.acheck(prompt=prompt)
        assert any(
            hit["prompt"] == prompt and hit["response"] == response for hit in hits
        ), f"scoped entry not returned; got {hits}"

    @pytest.mark.asyncio
    async def test_astore_with_per_entry_ttl_expires(
        self, langcache_with_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        """Async per-entry TTL should cause individual entries to expire."""

        prompt = f"Async per-entry TTL test [{scope}]"
        response = f"This async entry should expire quickly. [{scope}]"
        metadata = {"user_id": f"tenant_ttl_async_{scope}"}

        entry_id = await langcache_with_attrs.astore(
            prompt=prompt,
            response=response,
            metadata=metadata,
            ttl=5,
        )
        assert entry_id

        hits = await langcache_with_attrs.acheck(prompt=prompt, attributes=metadata)
        assert any(
            hit["response"] == response for hit in hits
        ), f"entry not retrievable before its TTL elapsed; got {hits}"

        await asyncio.sleep(6)

        hits_after_ttl = await langcache_with_attrs.acheck(
            prompt=prompt,
            attributes=metadata,
            num_results=5,
        )
        assert not any(hit["response"] == response for hit in hits_after_ttl)

    def test_store_with_metadata_and_check_with_attributes(
        self, langcache_with_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        prompt = f"Explain Redis search. [{scope}]"
        response = f"Redis provides full-text search via Redis Search. [{scope}]"
        # Use attribute names that are actually configured on this cache.
        metadata = {"user_id": f"tenant_a_{scope}"}

        entry_id = langcache_with_attrs.store(
            prompt=prompt,
            response=response,
            metadata=metadata,
            ttl=TEST_ENTRY_TTL,
        )
        assert entry_id

        hits = langcache_with_attrs.check(
            prompt=prompt,
            attributes=metadata,
            num_results=3,
        )
        assert any(
            hit["response"] == response for hit in hits
        ), f"attribute-filtered read did not return the scoped entry; got {hits}"

    def test_delete_by_id_and_by_attributes(
        self, langcache_with_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        prompt = f"Delete by id [{scope}]"
        response = f"Entry to delete by id. [{scope}]"
        metadata = {"user_id": f"tenant_delete_{scope}"}

        entry_id = langcache_with_attrs.store(
            prompt=prompt,
            response=response,
            metadata=metadata,
            ttl=TEST_ENTRY_TTL,
        )
        assert entry_id

        hits = langcache_with_attrs.check(
            prompt=prompt, attributes=metadata, num_results=5
        )
        assert any(hit["entry_id"] == entry_id for hit in hits)

        # delete by id
        langcache_with_attrs.delete_by_id(entry_id)
        hits_after_id_delete = langcache_with_attrs.check(
            prompt=prompt, attributes=metadata, num_results=5
        )
        assert not any(hit["entry_id"] == entry_id for hit in hits_after_id_delete)

        # store multiple entries and delete by attributes
        for i in range(3):
            langcache_with_attrs.store(
                prompt=f"{prompt} {i}",
                response=f"{response} {i}",
                metadata=metadata,
                ttl=TEST_ENTRY_TTL,
            )

        delete_result = langcache_with_attrs.delete_by_attributes(attributes=metadata)
        assert isinstance(delete_result, dict)
        # The attribute value is scope-unique, so all three stored entries must
        # be deleted -- not merely one of them.
        assert delete_result.get("deleted_entries_count", 0) >= 3

    @pytest.mark.asyncio
    async def test_async_delete_variants(
        self, langcache_with_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        prompt = f"Async delete by attributes [{scope}]"
        response = f"Async delete candidate [{scope}]"
        metadata = {"user_id": f"tenant_async_{scope}"}

        entry_id = await langcache_with_attrs.astore(
            prompt=prompt,
            response=response,
            metadata=metadata,
            ttl=TEST_ENTRY_TTL,
        )
        assert entry_id

        hits = await langcache_with_attrs.acheck(prompt=prompt, attributes=metadata)
        assert hits

        await langcache_with_attrs.adelete_by_id(entry_id)
        hits_after_id_delete = await langcache_with_attrs.acheck(
            prompt=prompt, attributes=metadata
        )
        assert not any(hit["entry_id"] == entry_id for hit in hits_after_id_delete)

        for i in range(2):
            await langcache_with_attrs.astore(
                prompt=f"{prompt} {i}",
                response=f"{response} {i}",
                metadata=metadata,
                ttl=TEST_ENTRY_TTL,
            )

        delete_result = await langcache_with_attrs.adelete_by_attributes(
            attributes=metadata
        )
        assert isinstance(delete_result, dict)
        assert delete_result.get("deleted_entries_count", 0) >= 2

    def test_attribute_value_with_comma_and_slash_is_encoded_for_llm_string(
        self, langcache_with_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        """llm_string attribute values with commas/slashes are client-encoded."""

        prompt = f"Attribute encoding for llm_string [{scope}]"
        response = f"Response for encoded llm_string. [{scope}]"

        raw_llm_string = f"tenant,with/slash_{scope}"
        entry_id = langcache_with_attrs.store(
            prompt=prompt,
            response=response,
            metadata={"llm_string": raw_llm_string},
            ttl=TEST_ENTRY_TTL,
        )
        assert entry_id

        # When we search using the *raw* llm_string value, the client should
        # transparently encode it before sending it to LangCache.
        hits = langcache_with_attrs.check(
            prompt=prompt,
            attributes={"llm_string": raw_llm_string},
            num_results=3,
        )
        # One hit must match on both counts: the response, and the metadata
        # round-tripped back to its original value (the client handles
        # encoding/decoding around the LangCache API).
        assert any(
            hit["response"] == response
            and hit.get("metadata", {}).get("llm_string") == raw_llm_string
            for hit in hits
        ), f"encoded llm_string did not round-trip; got {hits}"

    def test_attribute_value_with_all_tokenizer_separators_round_trip_and_filter(
        self, langcache_with_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        """All tokenizer separator characters should round-trip via filters.

        This exercises the set of punctuation described in the underlying
        Redis Search text-field tokenization docs to ensure that our
        client-side encoding/decoding and LangCache's attribute handling
        together can store and filter on values containing these characters.
        """

        separators = ",.<>{}[]\"':;!@#$%^&*()-+=~"
        raw_llm_string = f"tenant {separators} value {scope}"

        prompt = f"Attribute encoding for all tokenizer separators [{scope}]"
        response = f"Response for all tokenizer separators. [{scope}]"

        entry_id = langcache_with_attrs.store(
            prompt=prompt,
            response=response,
            metadata={"llm_string": raw_llm_string},
            ttl=TEST_ENTRY_TTL,
        )
        assert entry_id

        hits = langcache_with_attrs.check(
            prompt=prompt,
            attributes={"llm_string": raw_llm_string},
            num_results=5,
        )

        assert hits, "No hits returned for llm_string value with separators"
        assert any(
            hit.get("prompt") == prompt
            and hit.get("response") == response
            and hit.get("metadata", {}).get("llm_string") == raw_llm_string
            for hit in hits
        )

    @pytest.mark.parametrize(
        "raw_value",
        [
            r"tenant\\with\\backslash",
            "tenant?with?question",
        ],
    )
    def test_attribute_values_with_special_chars_round_trip_and_filter(
        self,
        langcache_with_attrs: LangCacheSemanticCache,
        raw_value: str,
        scope: str,
    ) -> None:
        """Backslash and question-mark values should round-trip via filters.

        These values previously failed attribute filtering on this LangCache
        instance; with URL-style percent encoding they should now be
        filterable and round-trip correctly.
        """

        # Joined with an underscore: unlike "-", it is not a text separator and
        # survives percent-encoding untouched, so scoping adds no character
        # beyond the ones this test exists to pin down.
        raw_value = f"{raw_value}_{scope}"
        prompt = f"Special chars attribute {raw_value}"
        response = f"Response for {raw_value}"

        entry_id = langcache_with_attrs.store(
            prompt=prompt,
            response=response,
            metadata={"llm_string": raw_value},
            ttl=TEST_ENTRY_TTL,
        )
        assert entry_id

        hits = langcache_with_attrs.check(
            prompt=prompt,
            attributes={"llm_string": raw_value},
            num_results=5,
        )

        # Look for a matching hit for this prompt/response/metadata triple.
        match_found = any(
            hit.get("prompt") == prompt
            and hit.get("response") == response
            and hit.get("metadata", {}).get("llm_string") == raw_value
            for hit in hits
        )

        assert match_found, (
            "Expected llm_string value to be filterable, but no matching "
            f"hit was found: {raw_value!r}"
        )


@pytest.mark.requires_api_keys
class TestLangCacheSemanticCacheIntegrationWithoutAttributes:
    def test_error_on_store_with_metadata_when_no_attributes_configured(
        self, langcache_no_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        prompt = f"Attributes not configured [{scope}]"
        response = "This should fail due to missing attributes configuration."

        # Scoped and TTL'd even though the store is expected to raise: if this
        # cache is ever given attributes, the write would start succeeding.
        with pytest.raises(RuntimeError) as exc:
            langcache_no_attrs.store(
                prompt=prompt,
                response=response,
                metadata={"tenant": f"tenant_without_attrs_{scope}"},
                ttl=TEST_ENTRY_TTL,
            )

        assert "attributes are not configured for this cache" in str(exc.value).lower()

    def test_error_on_check_with_attributes_when_no_attributes_configured(
        self, langcache_no_attrs: LangCacheSemanticCache
    ) -> None:
        prompt = "Attributes not configured on check"

        with pytest.raises(RuntimeError) as exc:
            langcache_no_attrs.check(
                prompt=prompt,
                attributes={"tenant": "tenant_without_attrs"},
            )

        assert "attributes are not configured for this cache" in str(exc.value).lower()

    def test_basic_store_and_check_works_without_attributes(
        self, langcache_no_attrs: LangCacheSemanticCache, scope: str
    ) -> None:
        prompt = f"Plain cache without attributes [{scope}]"
        response = f"This should be cached successfully. [{scope}]"

        entry_id = langcache_no_attrs.store(
            prompt=prompt, response=response, ttl=TEST_ENTRY_TTL
        )
        assert entry_id

        # This cache has no attributes configured, so a scoped filter is not
        # available here -- the unique prompt and the exact search strategy are
        # the only isolation this test can get.
        hits = langcache_no_attrs.check(prompt=prompt)
        assert any(
            hit["response"] == response for hit in hits
        ), f"scoped entry not returned; got {hits}"
