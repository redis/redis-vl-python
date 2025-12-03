import logging


def get_logger(name, log_level="info", fmt=None):
    """Return a logger instance."""

    # Use file name if logger is in debug mode
    name = "RedisVL" if log_level == "debug" else name

    logger = logging.getLogger(name)

    # Add a NullHandler to loggers to avoid "no handler found" warnings
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
