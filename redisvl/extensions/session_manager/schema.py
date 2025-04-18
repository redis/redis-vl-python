"""
RedisVL Session Manager Schema (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.message_history.schema` instead.
"""

import warnings

from redisvl.extensions.message_history.schema import (
    ChatMessage,
    SemanticMessageHistorySchema,
    MessageHistorySchema,
)

warnings.warn(
    "Importing from redisvl.extensions.session_manager.schema is deprecated. "
    "Please import from redisvl.extensions.message_history.schema instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ChatMessage",
    "MessageHistory",
    "SemanticMessageHistory",
]
