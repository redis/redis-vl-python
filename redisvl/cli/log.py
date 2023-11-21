import logging
import sys

import coloredlogs

# constants for logging
coloredlogs.DEFAULT_DATE_FORMAT = "%H:%M:%S"
coloredlogs.DEFAULT_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s   %(message)s"


def get_logger(name, log_level="info", fmt=None):
    """Return a logger instance."""

    # Use file name if logger is in debug mode
    name = "RedisVL" if log_level == "debug" else name

    logger = logging.getLogger(name)
    coloredlogs.install(level=log_level, logger=logger, fmt=fmt, stream=sys.stdout)
    return logger
