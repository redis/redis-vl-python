"""
RedisVL Semantic Session Manager (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.semantic_history` instead.
"""

import warnings

from redisvl.extensions.message_history.semantic_history import SemanticHistory

warnings.warn(
    "Importing from redisvl.extensions.session_manger.semantic_session is deprecated. "
    "Please import from redisvl.extensions.semantic_history instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SemanticHistory"]
