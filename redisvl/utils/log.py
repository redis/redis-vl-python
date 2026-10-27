import logging


def get_logger(name, log_level="info", fmt=None):
    """Return a logger instance.

    No handler is attached. A ``NullHandler`` counts as a handler and stops
    Python from sending WARNING and above to ``logging.lastResort`` (stderr)
    when the application has not configured logging.
    """

    # Use file name if logger is in debug mode
    name = "RedisVL" if log_level == "debug" else name

    return logging.getLogger(name)
