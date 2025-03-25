import logging
import sys

import coloredlogs

coloredlogs.DEFAULT_DATE_FORMAT = "%H:%M:%S"
coloredlogs.DEFAULT_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s   %(message)s"


def get_logger(name, log_level="info", fmt=None):
    """Return a logger instance."""

    # Use file name if logger is in debug mode
    name = "RedisVL" if log_level == "debug" else name

    logger = logging.getLogger(name)

    # Only configure this specific logger, not the root logger
    # Check if the logger already has handlers to respect existing configuration
    if not logger.handlers:
        coloredlogs.install(
            level=log_level,
            logger=logger,  # Pass the specific logger
            fmt=fmt,
            stream=sys.stdout,
            isatty=True,  # Only use colors when supported
            reconfigure=False,  # Don't reconfigure existing loggers
        )
    return logger
