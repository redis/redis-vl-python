"""
RedisVL Session Manager Extensions (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.message_history` instead.
"""

import warnings

from redisvl.extensions.session_manager.base_session import BaseSessionManager
from redisvl.extensions.session_manager.semantic_session import SemanticSessionManager
from redisvl.extensions.session_manager.standard_session import StandardSessionManager

warnings.warn(
    "Importing from redisvl.extensions.session_manager is deprecated. "
    "StandardSessionManager has been renamed to MessageHistory. "
    "SemanticSessionManager has been renamed to SemanticMessageHistory. "
    "Please import from redisvl.extensions.message_history instead.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["BaseSessionManager", "StandardSessionManager", "SemanticSessionManager"]
