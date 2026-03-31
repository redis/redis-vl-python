from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from redis.commands.search.query import Query

from redisvl.index import AsyncSearchIndex
from redisvl.migration.models import (
    MigrationPlan,
    MigrationValidation,
    QueryCheckResult,
)
from redisvl.migration.utils import load_yaml, schemas_equal
from redisvl.types import AsyncRedisClient


class AsyncMigrationValidator:
    """Async migration validator for post-migration checks.

    This is the async version of MigrationValidator. It uses AsyncSearchIndex
    and async Redis operations for better performance.
    """

    async def validate(
        self,
        plan: MigrationPlan,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedisClient] = None,
        query_check_file: Optional[str] = None,
    ) -> tuple[MigrationValidation, Dict[str, Any], float]:
        started = time.perf_counter()
        target_index = await AsyncSearchIndex.from_existing(
            plan.merged_target_schema["index"]["name"],
            redis_url=redis_url,
            redis_client=redis_client,
        )
        target_info = await target_index.info()
        validation = MigrationValidation()

        live_schema = target_index.schema.to_dict()
        # Exclude query-time and creation-hint attributes (ef_runtime, epsilon,
        # initial_cap, phonetic_matcher) that are not part of index structure
        # validation. Confirmed by RediSearch team as not relevant for this check.
        validation.schema_match = schemas_equal(
            live_schema, plan.merged_target_schema, strip_excluded=True
        )

        source_num_docs = int(plan.source.stats_snapshot.get("num_docs", 0) or 0)
        target_num_docs = int(target_info.get("num_docs", 0) or 0)
        validation.doc_count_match = source_num_docs == target_num_docs

        source_failures = int(
            plan.source.stats_snapshot.get("hash_indexing_failures", 0) or 0
        )
        target_failures = int(target_info.get("hash_indexing_failures", 0) or 0)
        validation.indexing_failures_delta = target_failures - source_failures

        key_sample = plan.source.keyspace.key_sample
        client = target_index.client
        if not key_sample:
            validation.key_sample_exists = True
        elif client is None:
            validation.key_sample_exists = False
            validation.errors.append("Failed to get Redis client for key sample check")
        else:
            # Handle prefix change: transform key_sample to use new prefix
            keys_to_check = key_sample
            if plan.rename_operations.change_prefix:
                old_prefix = plan.source.keyspace.prefixes[0]
                new_prefix = plan.rename_operations.change_prefix
                keys_to_check = [
                    new_prefix + k[len(old_prefix) :] if k.startswith(old_prefix) else k
                    for k in key_sample
                ]
            existing_count = await client.exists(*keys_to_check)
            validation.key_sample_exists = existing_count == len(keys_to_check)

        # Run automatic functional checks (always)
        functional_checks = await self._run_functional_checks(
            target_index, source_num_docs
        )
        validation.query_checks.extend(functional_checks)

        # Run user-provided query checks (if file provided)
        if query_check_file:
            user_checks = await self._run_query_checks(target_index, query_check_file)
            validation.query_checks.extend(user_checks)

        if not validation.schema_match:
            validation.errors.append("Live schema does not match merged_target_schema.")
        if not validation.doc_count_match:
            validation.errors.append(
                "Live document count does not match source num_docs."
            )
        if validation.indexing_failures_delta != 0:
            validation.errors.append("Indexing failures increased during migration.")
        if not validation.key_sample_exists:
            validation.errors.append(
                "One or more sampled source keys is missing after migration."
            )
        if any(not query_check.passed for query_check in validation.query_checks):
            validation.errors.append("One or more query checks failed.")

        return validation, target_info, round(time.perf_counter() - started, 3)

    async def _run_query_checks(
        self,
        target_index: AsyncSearchIndex,
        query_check_file: str,
    ) -> list[QueryCheckResult]:
        query_checks = load_yaml(query_check_file)
        results: list[QueryCheckResult] = []

        for doc_id in query_checks.get("fetch_ids", []):
            fetched = await target_index.fetch(doc_id)
            results.append(
                QueryCheckResult(
                    name=f"fetch:{doc_id}",
                    passed=fetched is not None,
                    details=(
                        "Document fetched successfully"
                        if fetched is not None
                        else "Document not found"
                    ),
                )
            )

        client = target_index.client
        for key in query_checks.get("keys_exist", []):
            if client is None:
                results.append(
                    QueryCheckResult(
                        name=f"key:{key}",
                        passed=False,
                        details="Failed to get Redis client",
                    )
                )
            else:
                exists = bool(await client.exists(key))
                results.append(
                    QueryCheckResult(
                        name=f"key:{key}",
                        passed=exists,
                        details="Key exists" if exists else "Key not found",
                    )
                )

        return results

    async def _run_functional_checks(
        self, target_index: AsyncSearchIndex, expected_doc_count: int
    ) -> List[QueryCheckResult]:
        """Run automatic functional checks to verify the index is operational.

        These checks run automatically after every migration to prove the index
        actually works, not just that the schema looks correct.
        """
        results: List[QueryCheckResult] = []

        # Check 1: Wildcard search - proves the index responds and returns docs
        try:
            search_result = await target_index.search(Query("*").paging(0, 1))
            total_found = search_result.total
            passed = total_found == expected_doc_count
            results.append(
                QueryCheckResult(
                    name="functional:wildcard_search",
                    passed=passed,
                    details=(
                        f"Wildcard search returned {total_found} docs "
                        f"(expected {expected_doc_count})"
                    ),
                )
            )
        except Exception as e:
            results.append(
                QueryCheckResult(
                    name="functional:wildcard_search",
                    passed=False,
                    details=f"Wildcard search failed: {str(e)}",
                )
            )

        return results
