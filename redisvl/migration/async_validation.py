from __future__ import annotations

import time
from typing import Any, Dict, Optional

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
        validation.schema_match = schemas_equal(live_schema, plan.merged_target_schema)

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
            existing_count = await client.exists(*key_sample)
            validation.key_sample_exists = existing_count == len(key_sample)

        if query_check_file:
            validation.query_checks = await self._run_query_checks(
                target_index,
                query_check_file,
            )

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
                        if fetched
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
