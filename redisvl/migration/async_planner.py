from __future__ import annotations

from typing import Any, List, Optional

from redisvl.index import AsyncSearchIndex
from redisvl.migration.models import (
    KeyspaceSnapshot,
    MigrationPlan,
    SchemaPatch,
    SourceSnapshot,
)
from redisvl.migration.planner import MigrationPlanner
from redisvl.redis.connection import supports_svs_async
from redisvl.schema.schema import IndexSchema
from redisvl.types import AsyncRedisClient


class AsyncMigrationPlanner:
    """Async migration planner for document-preserving drop/recreate flows.

    This is the async version of MigrationPlanner. It uses AsyncSearchIndex
    and async Redis operations for better performance on large indexes.

    The classification logic, schema merging, and diff analysis are delegated
    to a sync MigrationPlanner instance (they are CPU-bound and don't need async).
    """

    def __init__(self, key_sample_limit: int = 10):
        self.key_sample_limit = key_sample_limit
        # Delegate to sync planner for CPU-bound operations
        self._sync_planner = MigrationPlanner(key_sample_limit=key_sample_limit)

    # Expose static methods from MigrationPlanner for convenience
    get_vector_datatype_changes = staticmethod(
        MigrationPlanner.get_vector_datatype_changes
    )

    async def create_plan(
        self,
        index_name: str,
        *,
        redis_url: Optional[str] = None,
        schema_patch_path: Optional[str] = None,
        target_schema_path: Optional[str] = None,
        redis_client: Optional[AsyncRedisClient] = None,
    ) -> MigrationPlan:
        if not schema_patch_path and not target_schema_path:
            raise ValueError(
                "Must provide either --schema-patch or --target-schema for migration planning"
            )
        if schema_patch_path and target_schema_path:
            raise ValueError(
                "Provide only one of --schema-patch or --target-schema for migration planning"
            )

        snapshot = await self.snapshot_source(
            index_name,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        source_schema = IndexSchema.from_dict(snapshot.schema_snapshot)

        if schema_patch_path:
            schema_patch = self._sync_planner.load_schema_patch(schema_patch_path)
        else:
            # target_schema_path is guaranteed to be not None here
            assert target_schema_path is not None
            schema_patch = self._sync_planner.normalize_target_schema_to_patch(
                source_schema, target_schema_path
            )

        return await self.create_plan_from_patch(
            index_name,
            schema_patch=schema_patch,
            redis_url=redis_url,
            redis_client=redis_client,
            _snapshot=snapshot,
        )

    async def create_plan_from_patch(
        self,
        index_name: str,
        *,
        schema_patch: SchemaPatch,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedisClient] = None,
        _snapshot: Optional[Any] = None,
    ) -> MigrationPlan:
        if _snapshot is None:
            _snapshot = await self.snapshot_source(
                index_name,
                redis_url=redis_url,
                redis_client=redis_client,
            )
        snapshot = _snapshot
        source_schema = IndexSchema.from_dict(snapshot.schema_snapshot)
        merged_target_schema = self._sync_planner.merge_patch(
            source_schema, schema_patch
        )

        # Extract rename operations first
        rename_operations, rename_warnings = (
            self._sync_planner._extract_rename_operations(source_schema, schema_patch)
        )

        # Classify diff with awareness of rename operations
        diff_classification = self._sync_planner.classify_diff(
            source_schema, schema_patch, merged_target_schema, rename_operations
        )

        # Build warnings list
        warnings = ["Index downtime is required"]
        warnings.extend(rename_warnings)

        # Check for SVS-VAMANA in target schema and add appropriate warnings
        svs_warnings = await self._check_svs_vamana_requirements(
            merged_target_schema,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        warnings.extend(svs_warnings)

        return MigrationPlan(
            source=snapshot,
            requested_changes=schema_patch.model_dump(exclude_none=True),
            merged_target_schema=merged_target_schema.to_dict(),
            diff_classification=diff_classification,
            rename_operations=rename_operations,
            warnings=warnings,
        )

    async def _check_svs_vamana_requirements(
        self,
        target_schema: IndexSchema,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedisClient] = None,
    ) -> List[str]:
        """Async version: Check SVS-VAMANA requirements and return warnings."""
        warnings: List[str] = []
        target_dict = target_schema.to_dict()

        # Check if any vector field uses SVS-VAMANA
        uses_svs = False
        uses_compression = False
        compression_type = None

        for field in target_dict.get("fields", []):
            if field.get("type") != "vector":
                continue
            attrs = field.get("attrs", {})
            algo = attrs.get("algorithm", "").upper()
            if algo == "SVS-VAMANA":
                uses_svs = True
                compression = attrs.get("compression", "")
                if compression:
                    uses_compression = True
                    compression_type = compression

        if not uses_svs:
            return warnings

        # Check Redis version support
        try:
            if redis_client:
                client = redis_client
            elif redis_url:
                from redis.asyncio import Redis

                client = Redis.from_url(redis_url)
            else:
                client = None

            if client and not await supports_svs_async(client):
                warnings.append(
                    "SVS-VAMANA requires Redis >= 8.2.0 and Redis Search >= 2.8.10. "
                    "The target Redis instance may not support this algorithm. "
                    "Migration will fail at apply time if requirements are not met."
                )
        except Exception:
            warnings.append(
                "SVS-VAMANA requires Redis >= 8.2.0 and Redis Search >= 2.8.10. "
                "Verify your Redis instance supports this algorithm before applying."
            )

        # Intel hardware warning for compression
        if uses_compression:
            warnings.append(
                f"SVS-VAMANA with {compression_type} compression: "
                "LVQ and LeanVec optimizations require Intel hardware with AVX-512 support. "
                "On non-Intel platforms or Redis Open Source, these fall back to basic "
                "8-bit scalar quantization with reduced performance benefits."
            )
        else:
            warnings.append(
                "SVS-VAMANA: For optimal performance, Intel hardware with AVX-512 support "
                "is recommended. LVQ/LeanVec compression options provide additional memory "
                "savings on supported hardware."
            )

        return warnings

    async def snapshot_source(
        self,
        index_name: str,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedisClient] = None,
    ) -> SourceSnapshot:
        index = await AsyncSearchIndex.from_existing(
            index_name,
            redis_url=redis_url,
            redis_client=redis_client,
        )
        schema_dict = index.schema.to_dict()
        stats_snapshot = await index.info()
        prefixes = index.schema.index.prefix
        prefix_list = prefixes if isinstance(prefixes, list) else [prefixes]

        client = index.client
        if client is None:
            raise ValueError("Failed to get Redis client from index")

        return SourceSnapshot(
            index_name=index_name,
            schema_snapshot=schema_dict,
            stats_snapshot=stats_snapshot,
            keyspace=KeyspaceSnapshot(
                storage_type=index.schema.index.storage_type.value,
                prefixes=prefix_list,
                key_separator=index.schema.index.key_separator,
                key_sample=await self._async_sample_keys(
                    client=client,
                    prefixes=prefix_list,
                    key_separator=index.schema.index.key_separator,
                ),
            ),
        )

    async def _async_sample_keys(
        self, *, client: AsyncRedisClient, prefixes: List[str], key_separator: str
    ) -> List[str]:
        """Async version of _sample_keys."""
        key_sample: List[str] = []
        if self.key_sample_limit <= 0:
            return key_sample

        for prefix in prefixes:
            if len(key_sample) >= self.key_sample_limit:
                break
            match_pattern = (
                f"{prefix}*"
                if prefix.endswith(key_separator)
                else f"{prefix}{key_separator}*"
            )
            cursor: int = 0
            while True:
                cursor, keys = await client.scan(
                    cursor=cursor,
                    match=match_pattern,
                    count=max(self.key_sample_limit, 10),
                )
                for key in keys:
                    decoded_key = key.decode() if isinstance(key, bytes) else str(key)
                    if decoded_key not in key_sample:
                        key_sample.append(decoded_key)
                    if len(key_sample) >= self.key_sample_limit:
                        return key_sample
                if cursor == 0:
                    break
        return key_sample

    def write_plan(self, plan: MigrationPlan, plan_out: str) -> None:
        """Delegate to sync planner for file I/O."""
        self._sync_planner.write_plan(plan, plan_out)
