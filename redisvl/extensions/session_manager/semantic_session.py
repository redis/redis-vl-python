"""
RedisVL Semantic Session Manager (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.semantic_history` instead.
"""

import warnings

from redisvl.extensions.message_history.semantic_history import SemanticMessageHistory

warnings.warn(
    "Importing from redisvl.extensions.session_manger.semantic_session is deprecated. "
    "SemanticSessionManager has been renamed to SemanticMessageHistory. "
    "Please import SemanticMessageHistory from redisvl.extensions.semantic_history instead.",
    DeprecationWarning,
    stacklevel=2,
)


class SemanticSessionManager(SemanticMessageHistory):
    # keep for backwards compatibility
    pass


__all__ = ["SemanticSessionManager"]
