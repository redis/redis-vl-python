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
        if _has_application_handlers(record.name):
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


def _has_application_handlers(logger_name: str) -> bool:
    """True when the host app attached handlers outside RedisVL's default."""
    root = logging.getLogger()
    if root.handlers:
        return True
    parts = logger_name.split(".")
    for i in range(len(parts), 0, -1):
        for handler in logging.getLogger(".".join(parts[:i])).handlers:
            if not isinstance(handler, _DefaultStderrHandler):
                return True
    return False


def get_logger(name, log_level="info", fmt=None):
    """Return a logger instance."""

    # Use file name if logger is in debug mode
    name = "RedisVL" if log_level == "debug" else name

    logger = logging.getLogger(name)
    if not any(isinstance(h, _DefaultStderrHandler) for h in logger.handlers):
        logger.addHandler(_DefaultStderrHandler())
    return logger
