"""
RedisVL Standard Session Manager (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.base_history` instead.
"""

import warnings

from redisvl.extensions.message_history.base_history import BaseMessageHistory

warnings.warn(
    "Importing from redisvl.extensions.session_manager.base_session is deprecated. "
    "BaseSessionManager has been renamed to BaseMessageHistory. "
    "Please import BaseMessageHistory from redisvl.extensions.base_history instead.",
    DeprecationWarning,
    stacklevel=2,
)


class BaseSessionManager(BaseMessageHistory):
    # keep for backward compatibility
    pass


__all__ = ["BaseSessionManager"]
