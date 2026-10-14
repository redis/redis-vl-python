"""Code tools: async Python functions registered as MCP tools.

A code tool is an async function whose first parameter is a
:class:`~redisvl.mcp.context.ToolContext` and whose remaining, annotated
parameters are the arguments the model supplies. Its docstring (or an explicit
description) is what the model reads when choosing a tool.

Everything that makes a built-in safe is applied around the function rather
than left to its author: the read or write scope and any per-tool scope are
checked before it runs, the context it receives can only reach Redis through a
guarded, allow-listed handle, the whole call is bounded by the binding's
request timeout, and any exception it raises is mapped to the stable MCP error
contract instead of leaking a traceback.
"""

import asyncio
import importlib
import inspect
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from redisvl.mcp.auth import ensure_read_scope, ensure_tool_scope, ensure_write_scope
from redisvl.mcp.config import validate_tool_name
from redisvl.mcp.context import ToolContext
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError, map_exception

# Where `tool` leaves a function's spec for the loader to find. An attribute on
# the function rather than a module-level registry, so importing a module
# registers nothing anywhere: two servers in one process, and a test that
# imports a tool module, stay independent of each other.
_SPEC_ATTRIBUTE = "__redisvl_mcp_tool__"

_ToolFn = TypeVar("_ToolFn", bound=Callable[..., Any])

_ACCEPTED_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


@dataclass(frozen=True)
class CodeToolSpec:
    """One validated code tool, ready to register against a binding."""

    fn: Callable[..., Any]
    name: str
    description: str
    index: str | None
    writes: bool
    scope: str | None
    signature: inspect.Signature
    annotations: dict[str, Any]


def _qualified(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', fn)}"


def _tool_signature(
    fn: Callable[..., Any],
) -> tuple[inspect.Signature, dict[str, Any]]:
    """Return the model-facing signature: the function's, minus the context."""
    where = _qualified(fn)
    try:
        # Resolved rather than read raw, so a module using postponed
        # annotations is checked against the real types.
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:
        raise TypeError(
            f"code tool {where} has annotations that cannot be resolved: {exc}"
        ) from exc

    parameters = list(inspect.signature(fn).parameters.values())
    # Positional, because the wrapper passes the context positionally; a
    # keyword-only one would be accepted here and fail on every call.
    if (
        not parameters
        or hints.get(parameters[0].name) is not ToolContext
        or parameters[0].kind
        not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ):
        raise TypeError(
            f"code tool {where} must take a ToolContext as its first, positional "
            "parameter"
        )

    exposed = []
    annotations: dict[str, Any] = {}
    for parameter in parameters[1:]:
        if parameter.kind not in _ACCEPTED_KINDS:
            raise TypeError(
                f"code tool {where} parameter '{parameter.name}' must be a plain "
                "named parameter; *args, **kwargs and positional-only "
                "parameters cannot be offered to a model"
            )
        if parameter.name not in hints:
            raise TypeError(
                f"code tool {where} parameter '{parameter.name}' needs a type "
                "annotation; it is what the tool's input schema is built from"
            )
        annotation = hints[parameter.name]
        # Keyword-only, because the wrapper is always called with keyword
        # arguments decoded from the model's JSON.
        exposed.append(
            parameter.replace(
                kind=inspect.Parameter.KEYWORD_ONLY, annotation=annotation
            )
        )
        annotations[parameter.name] = annotation

    return_annotation = hints.get("return", inspect.Signature.empty)
    if return_annotation is not inspect.Signature.empty:
        annotations["return"] = return_annotation
    return (
        inspect.Signature(exposed, return_annotation=return_annotation),
        annotations,
    )


def build_code_tool_spec(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    namespace: str | None = None,
    index: str | None = None,
    writes: bool = False,
    scope: str | None = None,
    description: str | None = None,
) -> CodeToolSpec:
    """Validate ``fn`` as a code tool and return its spec.

    Everything checkable without a running server is checked here, so a
    malformed tool fails where it is declared rather than at startup.
    Whether ``index`` names a real binding, and whether a writing tool is
    allowed on it, depends on the server's config and is checked at startup.
    """
    where = _qualified(fn)
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"code tool {where} must be an async function")

    signature, annotations = _tool_signature(fn)

    resolved_description = (
        description if description is not None else inspect.getdoc(fn)
    )
    if not resolved_description or not resolved_description.strip():
        raise ValueError(
            f"code tool {where} needs a description: give it a docstring or pass "
            "description=; it is what the model chooses tools by"
        )

    resolved_name = name if name is not None else fn.__name__.replace("_", "-")
    if namespace is not None:
        resolved_name = f"{namespace}-{resolved_name}"
    validate_tool_name(resolved_name, kind=f"code tool {where}")

    if scope is not None and not scope.strip():
        raise ValueError(f"code tool {where} scope must not be blank")

    return CodeToolSpec(
        fn=fn,
        name=resolved_name,
        description=resolved_description.strip(),
        index=index,
        writes=writes,
        scope=scope,
        signature=signature,
        annotations=annotations,
    )


def register_code_tool(server: Any, spec: CodeToolSpec, binding_id: str) -> None:
    """Register one code tool against ``binding_id``."""

    async def code_tool(**kwargs: Any) -> Any:
        # Before anything else, so a caller without the scope never reaches the
        # binding or the function. The tool's own scope is required on top of
        # the server-wide one, so it can only make a tool stricter.
        if spec.writes:
            ensure_write_scope(server)
        else:
            ensure_read_scope(server)
        ensure_tool_scope(server, spec.scope)

        try:
            runtime = server.resolve_binding(binding_id)
            if spec.writes and runtime.effective_read_only:
                # Startup refuses this pairing, but the binding is re-read on
                # every start while tools register once, so a restart against
                # a config that made the index read-only lands here.
                raise RedisVLMCPError(
                    f"index '{binding_id}' is read-only",
                    code=MCPErrorCode.FORBIDDEN,
                    retryable=False,
                )
            ctx = ToolContext.for_request(server, runtime, writes=spec.writes)
            timeout = runtime.binding.runtime.request_timeout_seconds
            try:
                # The handle bounds each Redis call; this bounds the tool's own
                # code too, so a tool cannot hold a request open indefinitely.
                return await asyncio.wait_for(spec.fn(ctx, **kwargs), timeout)
            except asyncio.TimeoutError as exc:
                raise RedisVLMCPError(
                    f"tool '{spec.name}' did not finish within {timeout:g}s",
                    code=MCPErrorCode.BACKEND_UNAVAILABLE,
                    retryable=True,
                ) from exc
        except RedisVLMCPError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc

    code_tool.__name__ = spec.name.replace("-", "_")
    code_tool.__doc__ = spec.description
    code_tool.__signature__ = spec.signature  # type: ignore[attr-defined]
    code_tool.__annotations__ = dict(spec.annotations)

    server.tool(name=spec.name, description=spec.description)(code_tool)


def code_tool(
    *,
    name: str | None = None,
    namespace: str | None = None,
    index: str | None = None,
    writes: bool = False,
    scope: str | None = None,
    description: str | None = None,
) -> Callable[[_ToolFn], _ToolFn]:
    """Mark an async function as a code tool for a server to load.

    Takes the same arguments as ``RedisVLMCPServer.code_tool``. The function
    is checked here and returned unchanged; it is registered only on a server
    whose config lists its module under ``server.code_tool_modules``.
    """

    def decorator(fn: _ToolFn) -> _ToolFn:
        spec = build_code_tool_spec(
            fn,
            name=name,
            namespace=namespace,
            index=index,
            writes=writes,
            scope=scope,
            description=description,
        )
        setattr(fn, _SPEC_ATTRIBUTE, spec)
        return fn

    return decorator


def load_code_tools(modules: Iterable[str]) -> list[CodeToolSpec]:
    """Import each module and return the code tools it defines, in order.

    A module that cannot be imported, or that defines no tools, fails rather
    than contributing nothing, so a typo in a module name is not silently a
    server with fewer tools. A tool one module re-exports from another counts
    once.
    """
    specs: list[CodeToolSpec] = []
    seen: set[Callable[..., Any]] = set()
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ValueError(
                f"server.code_tool_modules entry '{module_name}' could not be "
                f"imported: {exc}"
            ) from exc

        # Functions only, and read from their own __dict__: a module attribute
        # with a permissive __getattr__ (a lazy module proxy, a mock) would
        # otherwise answer for any name, or import something just to say no.
        found = [
            value
            for value in vars(module).values()
            if inspect.isfunction(value)
            and isinstance(value.__dict__.get(_SPEC_ATTRIBUTE), CodeToolSpec)
        ]
        if not found:
            raise ValueError(
                f"server.code_tool_modules entry '{module_name}' defines no code "
                "tools; mark each one with @code_tool"
            )
        for fn in found:
            spec = fn.__dict__[_SPEC_ATTRIBUTE]
            if spec.fn is not fn:
                # functools.wraps copies __dict__, so a decorator applied over
                # @code_tool carries the marker while the spec still calls the
                # inner function: the outer decorator would silently never run.
                raise ValueError(
                    f"server.code_tool_modules entry '{module_name}': "
                    f"'{fn.__qualname__}' wraps a code tool in another decorator; "
                    "apply @code_tool outermost"
                )
            if fn not in seen:
                seen.add(fn)
                specs.append(spec)
    return specs
