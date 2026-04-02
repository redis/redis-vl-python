from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from redis.commands.search.query import Query

from redisvl.index import SearchIndex
from redisvl.migration.models import (
    MigrationPlan,
    MigrationValidation,
    QueryCheckResult,
)
from redisvl.migration.utils import load_yaml, schemas_equal


class MigrationValidator:
    def validate(
        self,
        plan: MigrationPlan,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[Any] = None,
        query_check_file: Optional[str] = None,
    ) -> tuple[MigrationValidation, Dict[str, Any], float]:
        started = time.perf_counter()
        target_index = SearchIndex.from_existing(
            plan.merged_target_schema["index"]["name"],
            redis_url=redis_url,
            redis_client=redis_client,
        )
        target_info = target_index.info()
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
        if not key_sample:
            validation.key_sample_exists = True
        else:
            # Handle prefix change: transform key_sample to use new prefix.
            # Must match the executor's RENAME logic exactly:
            #   new_key = new_prefix + key[len(old_prefix):]
            keys_to_check = key_sample
            if plan.rename_operations.change_prefix is not None:
                old_prefix = plan.source.keyspace.prefixes[0]
                new_prefix = plan.rename_operations.change_prefix
                key_sep = plan.source.keyspace.key_separator
                # Normalize prefixes: strip trailing separator for consistent slicing
                old_base = old_prefix.rstrip(key_sep) if old_prefix else ""
                new_base = new_prefix.rstrip(key_sep) if new_prefix else ""
                keys_to_check = []
                for k in key_sample:
                    if old_base and k.startswith(old_base):
                        keys_to_check.append(new_base + k[len(old_base) :])
                    else:
                        keys_to_check.append(k)
            existing_count = target_index.client.exists(*keys_to_check)
            validation.key_sample_exists = existing_count == len(keys_to_check)

        # Run automatic functional checks (always)
        functional_checks = self._run_functional_checks(target_index, source_num_docs)
        validation.query_checks.extend(functional_checks)

        # Run user-provided query checks (if file provided)
        if query_check_file:
            user_checks = self._run_query_checks(target_index, query_check_file)
            validation.query_checks.extend(user_checks)

        if not validation.schema_match and plan.validation.require_schema_match:
            validation.errors.append("Live schema does not match merged_target_schema.")
        if not validation.doc_count_match and plan.validation.require_doc_count_match:
            validation.errors.append(
                "Live document count does not match source num_docs."
            )
        if validation.indexing_failures_delta > 0:
            validation.errors.append("Indexing failures increased during migration.")
        if not validation.key_sample_exists:
            validation.errors.append(
                "One or more sampled source keys is missing after migration."
            )
        if any(not query_check.passed for query_check in validation.query_checks):
            validation.errors.append("One or more query checks failed.")

        return validation, target_info, round(time.perf_counter() - started, 3)

    def _run_query_checks(
        self,
        target_index: SearchIndex,
        query_check_file: str,
    ) -> list[QueryCheckResult]:
        query_checks = load_yaml(query_check_file)
        results: list[QueryCheckResult] = []

        for doc_id in query_checks.get("fetch_ids", []):
            fetched = target_index.fetch(doc_id)
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

        for key in query_checks.get("keys_exist", []):
            client = target_index.client
            if client is None:
                raise ValueError("Redis client not connected")
            exists = bool(client.exists(key))
            results.append(
                QueryCheckResult(
                    name=f"key:{key}",
                    passed=exists,
                    details="Key exists" if exists else "Key not found",
                )
            )

        return results

    def _run_functional_checks(
        self, target_index: SearchIndex, expected_doc_count: int
    ) -> List[QueryCheckResult]:
        """Run automatic functional checks to verify the index is operational.

        These checks run automatically after every migration to prove the index
        actually works, not just that the schema looks correct.
        """
        results: List[QueryCheckResult] = []

        # Check 1: Wildcard search - proves the index responds and returns docs
        try:
            search_result = target_index.search(Query("*").paging(0, 1))
            total_found = search_result.total
            passed = total_found > 0
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
