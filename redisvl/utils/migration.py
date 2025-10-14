"""
Utilities for migrating Redis indices to SVS-VAMANA with compression.

This module provides tools to migrate existing FLAT or HNSW indices to
SVS-VAMANA indices with compression, enabling significant memory savings
while maintaining search quality.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from redisvl.exceptions import RedisModuleVersionError
from redisvl.redis.connection import supports_svs
from redisvl.redis.constants import SVS_MIN_REDIS_VERSION
from redisvl.schema import IndexSchema
from redisvl.utils.compression import CompressionAdvisor
from redisvl.utils.log import get_logger

# Avoid circular imports by using TYPE_CHECKING
if TYPE_CHECKING:
    from redisvl.index import AsyncSearchIndex, SearchIndex
    from redisvl.query import FilterQuery
    from redisvl.query.filter import FilterExpression

logger = get_logger(__name__)


class IndexMigrator:
    """Helper class to migrate indices to SVS-VAMANA with compression.

    This class provides utilities to migrate existing FLAT or HNSW vector indices
    to SVS-VAMANA indices with compression, enabling significant memory savings.

    Example:
        .. code-block:: python

            from redisvl.index import SearchIndex
            from redisvl.utils import IndexMigrator

            # Load existing index
            old_index = SearchIndex.from_existing("my_flat_index")

            # Migrate to SVS-VAMANA with LVQ compression
            new_index = IndexMigrator.migrate_to_svs(
                old_index,
                compression="LVQ4x4",
                batch_size=1000
            )

            print(f"Migrated {new_index.info()['num_docs']} documents")
    """

    @staticmethod
    def migrate_to_svs(
        old_index: "SearchIndex",
        new_index_name: Optional[str] = None,
        compression: Optional[str] = None,
        reduce: Optional[int] = None,
        batch_size: int = 1000,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> "SearchIndex":
        """Migrate an existing index to SVS-VAMANA with compression.

        This method creates a new SVS-VAMANA index and copies all data from the
        old index in batches. The old index is not modified or deleted.

        Args:
            old_index: The existing SearchIndex to migrate from.
            new_index_name: Name for the new index. If None, uses "{old_name}_svs".
            compression: Compression type (LVQ4, LVQ4x4, LVQ4x8, LVQ8, LeanVec4x8, LeanVec8x8).
                If None, uses CompressionAdvisor to recommend based on dimensions.
            reduce: Dimensionality reduction parameter for LeanVec compression.
                Required for LeanVec compression types.
            batch_size: Number of documents to migrate per batch. Default: 1000.
            overwrite: Whether to overwrite the new index if it exists. Default: False.
            progress_callback: Optional callback function(current, total) for progress tracking.

        Returns:
            SearchIndex: The new SVS-VAMANA index with migrated data.

        Raises:
            RedisModuleVersionError: If Redis version doesn't support SVS-VAMANA.
            ValueError: If the old index has no vector fields or invalid parameters.

        Example:
            .. code-block:: python

                def progress(current, total):
                    print(f"Migrated {current}/{total} documents")

                new_index = IndexMigrator.migrate_to_svs(
                    old_index,
                    compression="LVQ4x4",
                    batch_size=500,
                    progress_callback=progress
                )
        """
        # Import here to avoid circular imports
        from redisvl.index import SearchIndex
        from redisvl.query import FilterQuery
        from redisvl.query.filter import FilterExpression

        # Check SVS-VAMANA support
        if not supports_svs(old_index._redis_client):
            raise RedisModuleVersionError.for_svs_vamana(SVS_MIN_REDIS_VERSION)

        # Find vector fields in the old schema
        vector_fields = [
            (name, field)
            for name, field in old_index.schema.fields.items()
            if hasattr(field, "attrs") and hasattr(field.attrs, "algorithm")
        ]

        if not vector_fields:
            raise ValueError("Old index has no vector fields to migrate")

        # Create new schema based on old schema
        new_schema_dict = old_index.schema.to_dict()

        # Update index name
        if new_index_name is None:
            new_index_name = f"{old_index.name}_svs"
        new_schema_dict["index"]["name"] = new_index_name
        new_schema_dict["index"]["prefix"] = new_index_name

        # Update vector fields to use SVS-VAMANA
        for field_dict in new_schema_dict["fields"]:
            if field_dict["type"] == "vector":
                attrs = field_dict.get("attrs", {})
                dims = attrs.get("dims")

                if dims is None:
                    raise ValueError(f"Vector field '{field_dict['name']}' has no dims")

                # Use CompressionAdvisor if compression not specified
                if compression is None:
                    config = CompressionAdvisor.recommend(
                        dims=dims,
                        priority="balanced",
                        datatype=attrs.get("datatype", "float32"),
                    )
                    compression = config["compression"]
                    if "reduce" in config and reduce is None:
                        reduce = config["reduce"]
                    logger.info(
                        f"CompressionAdvisor recommended: {compression} "
                        f"(reduce={reduce}) for {dims} dims"
                    )

                # Update to SVS-VAMANA
                attrs["algorithm"] = "svs-vamana"
                attrs["compression"] = compression

                if reduce is not None:
                    attrs["reduce"] = reduce

                # Set default SVS parameters if not present
                if "graph_max_degree" not in attrs:
                    attrs["graph_max_degree"] = 40
                if "construction_window_size" not in attrs:
                    attrs["construction_window_size"] = 250
                if "search_window_size" not in attrs:
                    attrs["search_window_size"] = 20
                # Set a low training threshold for small datasets
                # Default is 10240, minimum is 1024 (DEFAULT_BLOCK_SIZE)
                if "training_threshold" not in attrs:
                    attrs["training_threshold"] = 1024

        # Create new index
        new_schema = IndexSchema.from_dict(new_schema_dict)
        new_index = SearchIndex(schema=new_schema, redis_client=old_index._redis_client)
        new_index.create(overwrite=overwrite)

        logger.info(f"Created new SVS-VAMANA index: {new_index_name}")

        # Get total document count
        old_info = old_index.info()
        total_docs = int(old_info.get("num_docs", 0))

        if total_docs == 0:
            logger.warning("Old index has no documents to migrate")
            return new_index

        logger.info(f"Migrating {total_docs} documents in batches of {batch_size}")

        # Migrate data in batches using pagination
        migrated_count = 0
        query = FilterQuery(
            filter_expression=FilterExpression("*"),
            return_fields=list(old_index.schema.fields.keys()),
        )

        for batch in old_index.paginate(query, page_size=batch_size):
            if batch:
                # The 'id' field contains the full Redis key (e.g., "prefix:ulid")
                # We need to preserve the document ID part for the new index
                batch_keys = []
                batch_docs = []

                for doc in batch:
                    # Get the full Redis key from the id field
                    full_key = doc.get("id", "")
                    # Extract the document ID (everything after the prefix)
                    # Split by the key separator and take the last part
                    doc_id = full_key.split(old_index.schema.index.key_separator)[-1]

                    # Create a copy of the document without the id field
                    # (the id field is metadata, not actual document data)
                    doc_copy = {k: v for k, v in doc.items() if k != "id"}

                    batch_keys.append(new_index.key(doc_id))
                    batch_docs.append(doc_copy)

                # Load batch to new index with explicit keys to preserve IDs
                new_index.load(batch_docs, keys=batch_keys)
                migrated_count += len(batch)

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(migrated_count, total_docs)

                logger.debug(f"Migrated {migrated_count}/{total_docs} documents")

        logger.info(f"Migration complete: {migrated_count} documents migrated")

        # Verify migration by checking Redis keys (not index count)
        # Note: SVS-VAMANA indices have a training_threshold (default 10240)
        # Documents are written to Redis but may not be indexed until threshold is reached
        new_info = new_index.info()
        new_doc_count = int(new_info.get("num_docs", 0))

        if new_doc_count != total_docs:
            # Check if documents exist in Redis even if not indexed yet
            client = new_index._redis_client
            actual_keys = client.keys(f"{new_index.schema.index.prefix}:*")
            actual_count = len(actual_keys)

            if actual_count == total_docs:
                logger.info(
                    f"Documents written to Redis: {actual_count}/{total_docs}. "
                    f"Index shows {new_doc_count} (may be below training_threshold)"
                )
            else:
                logger.warning(
                    f"Document count mismatch: expected {total_docs}, "
                    f"got {actual_count} in Redis, {new_doc_count} in index"
                )

        return new_index

    @staticmethod
    async def migrate_to_svs_async(
        old_index: "AsyncSearchIndex",
        new_index_name: Optional[str] = None,
        compression: Optional[str] = None,
        reduce: Optional[int] = None,
        batch_size: int = 1000,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> "AsyncSearchIndex":
        """Asynchronously migrate an existing index to SVS-VAMANA with compression.

        This is the async version of migrate_to_svs(). See migrate_to_svs() for
        detailed documentation.

        Args:
            old_index: The existing AsyncSearchIndex to migrate from.
            new_index_name: Name for the new index. If None, uses "{old_name}_svs".
            compression: Compression type. If None, uses CompressionAdvisor.
            reduce: Dimensionality reduction parameter for LeanVec.
            batch_size: Number of documents to migrate per batch. Default: 1000.
            overwrite: Whether to overwrite the new index if it exists. Default: False.
            progress_callback: Optional callback function(current, total) for progress.

        Returns:
            AsyncSearchIndex: The new SVS-VAMANA index with migrated data.

        Example:
            .. code-block:: python

                async def progress(current, total):
                    print(f"Migrated {current}/{total} documents")

                new_index = await IndexMigrator.migrate_to_svs_async(
                    old_index,
                    compression="LVQ4x4",
                    progress_callback=progress
                )
        """
        # Import here to avoid circular imports
        from redisvl.index import AsyncSearchIndex
        from redisvl.query import FilterQuery
        from redisvl.query.filter import FilterExpression
        from redisvl.redis.connection import supports_svs_async

        # Check SVS-VAMANA support
        client = await old_index._get_client()
        if not await supports_svs_async(client):
            raise RedisModuleVersionError.for_svs_vamana(SVS_MIN_REDIS_VERSION)

        # Find vector fields
        vector_fields = [
            (name, field)
            for name, field in old_index.schema.fields.items()
            if hasattr(field, "attrs") and hasattr(field.attrs, "algorithm")
        ]

        if not vector_fields:
            raise ValueError("Old index has no vector fields to migrate")

        # Create new schema
        new_schema_dict = old_index.schema.to_dict()

        if new_index_name is None:
            new_index_name = f"{old_index.name}_svs"
        new_schema_dict["index"]["name"] = new_index_name
        new_schema_dict["index"]["prefix"] = new_index_name

        # Update vector fields
        for field_dict in new_schema_dict["fields"]:
            if field_dict["type"] == "vector":
                attrs = field_dict.get("attrs", {})
                dims = attrs.get("dims")

                if dims is None:
                    raise ValueError(f"Vector field '{field_dict['name']}' has no dims")

                if compression is None:
                    config = CompressionAdvisor.recommend(
                        dims=dims,
                        priority="balanced",
                        datatype=attrs.get("datatype", "float32"),
                    )
                    compression = config["compression"]
                    if "reduce" in config and reduce is None:
                        reduce = config["reduce"]
                    logger.info(
                        f"CompressionAdvisor recommended: {compression} "
                        f"(reduce={reduce}) for {dims} dims"
                    )

                attrs["algorithm"] = "svs-vamana"
                attrs["compression"] = compression

                if reduce is not None:
                    attrs["reduce"] = reduce

                if "graph_max_degree" not in attrs:
                    attrs["graph_max_degree"] = 40
                if "construction_window_size" not in attrs:
                    attrs["construction_window_size"] = 250
                if "search_window_size" not in attrs:
                    attrs["search_window_size"] = 20
                if "training_threshold" not in attrs:
                    attrs["training_threshold"] = 1024

        # Create new index
        new_schema = IndexSchema.from_dict(new_schema_dict)
        new_index = AsyncSearchIndex(schema=new_schema, redis_client=client)
        await new_index.create(overwrite=overwrite)

        logger.info(f"Created new SVS-VAMANA index: {new_index_name}")

        # Get total document count
        old_info = await old_index.info()
        total_docs = int(old_info.get("num_docs", 0))

        if total_docs == 0:
            logger.warning("Old index has no documents to migrate")
            return new_index

        logger.info(f"Migrating {total_docs} documents in batches of {batch_size}")

        # Migrate data in batches
        migrated_count = 0
        query = FilterQuery(
            filter_expression=FilterExpression("*"),
            return_fields=list(old_index.schema.fields.keys()),
        )

        async for batch in old_index.paginate(query, page_size=batch_size):
            if batch:
                # Extract document IDs from full Redis keys
                batch_keys = []
                batch_docs = []

                for doc in batch:
                    full_key = doc.get("id", "")
                    doc_id = full_key.split(old_index.schema.index.key_separator)[-1]
                    doc_copy = {k: v for k, v in doc.items() if k != "id"}

                    batch_keys.append(new_index.key(doc_id))
                    batch_docs.append(doc_copy)

                # Load batch to new index with explicit keys
                await new_index.load(batch_docs, keys=batch_keys)
                migrated_count += len(batch)

                if progress_callback:
                    progress_callback(migrated_count, total_docs)

                logger.debug(f"Migrated {migrated_count}/{total_docs} documents")

        logger.info(f"Migration complete: {migrated_count} documents migrated")

        # Verify migration by checking Redis keys
        new_info = await new_index.info()
        new_doc_count = int(new_info.get("num_docs", 0))

        if new_doc_count != total_docs:
            # Check if documents exist in Redis even if not indexed yet
            actual_keys = await client.keys(f"{new_index.schema.index.prefix}:*")
            actual_count = len(actual_keys)

            if actual_count == total_docs:
                logger.info(
                    f"Documents written to Redis: {actual_count}/{total_docs}. "
                    f"Index shows {new_doc_count} (may be below training_threshold)"
                )
            else:
                logger.warning(
                    f"Document count mismatch: expected {total_docs}, "
                    f"got {actual_count} in Redis, {new_doc_count} in index"
                )

        return new_index
