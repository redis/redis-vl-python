"""
RedisVL Standard Session Manager (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.message_history` instead.
"""

import warnings

from redisvl.extensions.message_history.standard_history import MessageHistory

warnings.warn(
    "Importing from redisvl.extensions.session_manger.standard_session is deprecated. "
    "Please import from redisvl.extensions.message_history instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MessageHistory"]
