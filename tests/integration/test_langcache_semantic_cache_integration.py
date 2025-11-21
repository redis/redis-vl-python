"""Integration tests for LangCacheSemanticCache against the LangCache managed service.

These tests exercise the real LangCache API using two configured caches:
- One with attributes configured
- One without attributes configured

Env vars (loaded from .env locally, injected via CI):
- LANGCACHE_WITH_ATTRIBUTES_API_KEY
- LANGCACHE_WITH_ATTRIBUTES_CACHE_ID
- LANGCACHE_WITH_ATTRIBUTES_URL
- LANGCACHE_NO_ATTRIBUTES_API_KEY
- LANGCACHE_NO_ATTRIBUTES_CACHE_ID
- LANGCACHE_NO_ATTRIBUTES_URL
"""

import os
from typing import Dict

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


def _require_env_vars(var_names: tuple[str, ...]) -> Dict[str, str]:
    missing = [name for name in var_names if not os.getenv(name)]
    if missing:
        pytest.skip(
            f"Missing required LangCache env vars: {', '.join(missing)}. "
            "Set them locally (e.g., via .env) or in CI secrets to run these tests."
        )

    return {name: os.environ[name] for name in var_names}


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
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        prompt = "What is Redis?"
        response = "Redis is an in-memory data store."

        entry_id = langcache_with_attrs.store(prompt=prompt, response=response)
        assert entry_id

        hits = langcache_with_attrs.check(prompt=prompt, num_results=1)
        assert hits
        assert hits[0]["response"] == response
        assert hits[0]["prompt"] == prompt

    @pytest.mark.asyncio
    async def test_store_and_check_async(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        prompt = "What is Redis async?"
        response = "Redis is an in-memory data store (async)."

        entry_id = await langcache_with_attrs.astore(prompt=prompt, response=response)
        assert entry_id

        hits = await langcache_with_attrs.acheck(prompt=prompt, num_results=1)
        assert hits
        assert hits[0]["response"] == response
        assert hits[0]["prompt"] == prompt

    def test_store_with_metadata_and_check_with_attributes(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        prompt = "Explain Redis search."
        response = "Redis provides full-text search via RediSearch."
        # Use attribute names that are actually configured on this cache.
        metadata = {"user_id": "tenant_a"}

        entry_id = langcache_with_attrs.store(
            prompt=prompt,
            response=response,
            metadata=metadata,
        )
        assert entry_id

        hits = langcache_with_attrs.check(
            prompt=prompt,
            attributes={"user_id": "tenant_a"},
            num_results=3,
        )
        assert hits
        assert any(hit["response"] == response for hit in hits)

    def test_delete_and_clear_alias(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        """delete() and clear() should flush the whole cache."""

        prompt = "Delete me"
        response = "You won't see me again."

        langcache_with_attrs.store(prompt=prompt, response=response)
        hits_before = langcache_with_attrs.check(prompt=prompt, num_results=5)
        assert hits_before

        # delete() and clear() both flush the whole cache
        langcache_with_attrs.delete()
        hits_after_delete = langcache_with_attrs.check(prompt=prompt, num_results=5)

        # It is possible for other tests or data to exist; we only assert that
        # the original response is no longer present if any hits are returned.
        assert not any(hit["response"] == response for hit in hits_after_delete)

        langcache_with_attrs.store(prompt=prompt, response=response)
        langcache_with_attrs.clear()
        hits_after_clear = langcache_with_attrs.check(prompt=prompt, num_results=5)
        assert not any(hit["response"] == response for hit in hits_after_clear)

    def test_delete_by_id_and_by_attributes(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        prompt = "Delete by id"
        response = "Entry to delete by id."
        metadata = {"user_id": "tenant_delete"}

        entry_id = langcache_with_attrs.store(
            prompt=prompt,
            response=response,
            metadata=metadata,
        )
        assert entry_id

        hits = langcache_with_attrs.check(
            prompt=prompt, attributes=metadata, num_results=1
        )
        assert hits
        assert hits[0]["entry_id"] == entry_id

        # delete by id
        langcache_with_attrs.delete_by_id(entry_id)
        hits_after_id_delete = langcache_with_attrs.check(
            prompt=prompt, attributes=metadata, num_results=3
        )
        assert not any(hit["entry_id"] == entry_id for hit in hits_after_id_delete)

        # store multiple entries and delete by attributes
        for i in range(3):
            langcache_with_attrs.store(
                prompt=f"{prompt} {i}",
                response=f"{response} {i}",
                metadata=metadata,
            )

        delete_result = langcache_with_attrs.delete_by_attributes(attributes=metadata)
        assert isinstance(delete_result, dict)
        assert delete_result.get("deleted_entries_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_async_delete_variants(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        prompt = "Async delete by attributes"
        response = "Async delete candidate"
        metadata = {"user_id": "tenant_async"}

        entry_id = await langcache_with_attrs.astore(
            prompt=prompt,
            response=response,
            metadata=metadata,
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
            )

        delete_result = await langcache_with_attrs.adelete_by_attributes(
            attributes=metadata
        )
        assert isinstance(delete_result, dict)
        assert delete_result.get("deleted_entries_count", 0) >= 1

        # Finally, aclear() should flush the cache.
        await langcache_with_attrs.aclear()
        hits_after_clear = await langcache_with_attrs.acheck(
            prompt=prompt, num_results=5
        )
        assert not any(hit["response"] == response for hit in hits_after_clear)

    def test_attribute_value_with_comma_and_slash_is_encoded_for_llm_string(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        """llm_string attribute values with commas/slashes are client-encoded."""

        prompt = "Attribute encoding for llm_string"
        response = "Response for encoded llm_string."

        raw_llm_string = "tenant,with/slash"
        entry_id = langcache_with_attrs.store(
            prompt=prompt,
            response=response,
            metadata={"llm_string": raw_llm_string},
        )
        assert entry_id

        # When we search using the *raw* llm_string value, the client should
        # transparently encode it before sending it to LangCache.
        hits = langcache_with_attrs.check(
            prompt=prompt,
            attributes={"llm_string": raw_llm_string},
            num_results=3,
        )
        assert hits
        # Response must match, and metadata should contain the original value
        # (the client handles encoding/decoding around the LangCache API).
        assert any(hit["response"] == response for hit in hits)
        assert any(
            hit.get("metadata", {}).get("llm_string") == raw_llm_string for hit in hits
        )


@pytest.mark.requires_api_keys
class TestLangCacheSemanticCacheIntegrationWithoutAttributes:
    def test_error_on_store_with_metadata_when_no_attributes_configured(
        self, langcache_no_attrs: LangCacheSemanticCache
    ) -> None:
        prompt = "Attributes not configured"
        response = "This should fail due to missing attributes configuration."

        with pytest.raises(RuntimeError) as exc:
            langcache_no_attrs.store(
                prompt=prompt,
                response=response,
                metadata={"tenant": "tenant_without_attrs"},
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
        self, langcache_no_attrs: LangCacheSemanticCache
    ) -> None:
        prompt = "Plain cache without attributes"
        response = "This should be cached successfully."

        entry_id = langcache_no_attrs.store(prompt=prompt, response=response)
        assert entry_id

        hits = langcache_no_attrs.check(prompt=prompt)
        assert hits
        assert any(hit["response"] == response for hit in hits)
