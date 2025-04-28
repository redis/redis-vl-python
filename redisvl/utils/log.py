import logging
import sys


def get_logger(name, log_level="info", fmt=None):
    """Return a logger instance."""

    # Use file name if logger is in debug mode
    name = "RedisVL" if log_level == "debug" else name

    logger = logging.getLogger(name)

    # Only configure this specific logger, not the root logger
    # Check if the logger already has handlers to respect existing configuration
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s   %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
    return logger
