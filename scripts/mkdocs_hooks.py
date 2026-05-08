"""mkdocs hooks for the redisvl docs build.

Suppress noisy griffe diagnostics so `mkdocs build --strict` is not blocked
by docstring-quality nits in the source tree (parameter naming, missing
`**kwargs` annotations, malformed `Raises:` lines). Only ERROR-level griffe
output (genuine import or parse failures) bubbles up to mkdocs' strict mode.
"""

import logging

_QUIET_LOGGERS = (
    "griffe",
    "mkdocs.plugins.griffe",
    "mkdocs.plugins.mkdocstrings",
    "mkdocs.plugins.mkdocstrings_handlers.python",
    "mkdocs.plugins.mkdocstrings_handlers.python._internal",
    "mkdocs.plugins.mkdocstrings_handlers.python._internal.handler",
)


def _quiet():
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


_quiet()


def on_startup(*, command, dirty):  # noqa: ARG001
    _quiet()


def on_config(config, **_kwargs):
    _quiet()
    return config
