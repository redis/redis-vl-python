"""
RedisVL Standard Session Manager (Deprecated Path)

This module is kept for backward compatibility. Please use `redisvl.extensions.message_history` instead.
"""

import warnings

from redisvl.extensions.message_history.message_history import MessageHistory

warnings.warn(
    "Importing from redisvl.extensions.session_manger.standard_session is deprecated. "
    "StandardSessionManager has been renamed to MessageHistory. "
    "Please import MessageHistory from redisvl.extensions.message_history instead.",
    DeprecationWarning,
    stacklevel=2,
)


class StandardSessionManager(MessageHistory):
    # keep for backward compatibility
    pass


__all__ = ["StandardSessionManager"]
