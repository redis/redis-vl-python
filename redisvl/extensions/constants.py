"""
Constants used within the extension classes SemanticCache, BaseMessageHistory,
MessageHistory, SemanticMessageHistory and SemanticRouter.
These constants are also used within these classes' corresponding schemas.
"""

# BaseMessageHistory
ID_FIELD_NAME: str = "entry_id"
ROLE_FIELD_NAME: str = "role"
CONTENT_FIELD_NAME: str = "content"
TOOL_FIELD_NAME: str = "tool_call_id"
TIMESTAMP_FIELD_NAME: str = "timestamp"
SESSION_FIELD_NAME: str = "session_tag"

# SemanticMessageHistory
MESSAGE_VECTOR_FIELD_NAME: str = "vector_field"

# SemanticCache
REDIS_KEY_FIELD_NAME: str = "key"
ENTRY_ID_FIELD_NAME: str = "entry_id"
PROMPT_FIELD_NAME: str = "prompt"
RESPONSE_FIELD_NAME: str = "response"
CACHE_VECTOR_FIELD_NAME: str = "prompt_vector"
INSERTED_AT_FIELD_NAME: str = "inserted_at"
UPDATED_AT_FIELD_NAME: str = "updated_at"
METADATA_FIELD_NAME: str = "metadata"  # also used in MessageHistory

# EmbeddingsCache
TEXT_FIELD_NAME: str = "text"
MODEL_NAME_FIELD_NAME: str = "model_name"
EMBEDDING_FIELD_NAME: str = "embedding"
DIMENSIONS_FIELD_NAME: str = "dimensions"

# SemanticRouter
ROUTE_VECTOR_FIELD_NAME: str = "vector"

# Raised by every constructor that accepts both `create_index` and `overwrite`.
CREATE_INDEX_OVERWRITE_CONFLICT: str = (
    "create_index=False and overwrite=True contradict each other: overwrite asks "
    "RedisVL to drop and recreate the index, which it cannot do when it is told "
    "not to manage the index at all."
)

# Raised when an extension attached to an externally managed index is asked to
# perform an index-wide destructive operation.
EXTERNAL_INDEX_LIFECYCLE_CONFLICT: str = (
    "Cannot delete or clear an index when create_index=False because RedisVL "
    "does not manage that index's lifecycle. Use the externally managed "
    "provisioning path to perform index-wide destructive operations."
)
