import logging


class _DefaultStderrHandler(logging.Handler):
    """Emit WARNING+ when the application has not configured logging.

    ``NullHandler`` counts as a handler and prevents ``logging.lastResort``
    from running, which hid RedisVL warnings before #786. This handler forwards
    WARNING and above to stderr (via ``lastResort``) when the root logger has
    no handlers, and forwards to the root logger once logging is configured.
    Exception tracebacks are not written here so CLI code can log with
    ``exc_info`` for configured apps without duplicating tracebacks on stderr.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        root = logging.getLogger()
        if root.handlers:
            root.handle(record)
            return
        clean = logging.makeLogRecord(
            {
                **record.__dict__,
                "exc_info": None,
                "exc_text": None,
                "stack_info": None,
            }
        )
        logging.lastResort.handle(clean)


def get_logger(name, log_level="info", fmt=None):
    """Return a logger instance."""

    # Use file name if logger is in debug mode
    name = "RedisVL" if log_level == "debug" else name

    logger = logging.getLogger(name)
    if not any(isinstance(h, _DefaultStderrHandler) for h in logger.handlers):
        logger.addHandler(_DefaultStderrHandler())
        logger.propagate = False
    return logger
