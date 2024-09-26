"""
Constants used within the extension classes SemanticCache, BaseSessionManager,
StandardSessionManager,SemanticSessionManager and SemanticRouter.
These constants are also used within theses classes corresponding schema.
"""

# BaseSessionManager
ID_FIELD_NAME: str = "entry_id"
ROLE_FIELD_NAME: str = "role"
CONTENT_FIELD_NAME: str = "content"
TOOL_FIELD_NAME: str = "tool_call_id"
TIMESTAMP_FIELD_NAME: str = "timestamp"
SESSION_FIELD_NAME: str = "session_tag"

# SemanticSessionManager
SESSION_VECTOR_FIELD_NAME: str = "vector_field"

# SemanticCache
REDIS_KEY_FIELD_NAME: str = "key"
ENTRY_ID_FIELD_NAME: str = "entry_id"
PROMPT_FIELD_NAME: str = "prompt"
RESPONSE_FIELD_NAME: str = "response"
CACHE_VECTOR_FIELD_NAME: str = "prompt_vector"
INSERTED_AT_FIELD_NAME: str = "inserted_at"
UPDATED_AT_FIELD_NAME: str = "updated_at"
METADATA_FIELD_NAME: str = "metadata"

# SemanticRouter
ROUTE_VECTOR_FIELD_NAME: str = "vector"
