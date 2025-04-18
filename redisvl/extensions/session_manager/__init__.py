"""
RedisVL Session Manager Extensions (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.message_history` instead.
"""

import warnings

from redisvl.extensions.message_history.message_history import MessageHistory
from redisvl.extensions.message_history.schema import (
    ChatMessage,
    MessageHistorySchema,
    SemanticMessageHistorySchema,
)
from redisvl.extensions.message_history.semantic_message_history import (
    SemanticMessageHistory,
)

warnings.warn(
    "Importing from redisvl.extensions.session_manager is deprecated. "
    "Please import from redisvl.extensions.message_history instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "MessageHistory",
    "SemanticMessageHistory",
    "ChatMessage",
    "MessageHistorySchema",
    "SemanticMessageHistorySchema",
]
