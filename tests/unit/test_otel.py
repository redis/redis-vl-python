"""RedisVL's OpenTelemetry plumbing: scope, off switch, and running with only
the API or with no OpenTelemetry at all."""

import importlib.metadata
import json
import subprocess
import sys
import textwrap

import pytest
from packaging.requirements import Requirement

from redisvl.utils import otel
from redisvl.version import __version__


def _metric_scopes(reader):
    """Map each exported metric name to its instrumentation scope."""
    data = reader.get_metrics_data()
    return {
        metric.name: scope_metrics.scope
        for resource_metrics in (data.resource_metrics if data else [])
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }


def test_telemetry_is_scoped_to_redisvl_and_its_version(otel_sdk):
    spans, reader = otel_sdk
    with otel.get_tracer().start_as_current_span("probe"):
        pass
    otel.get_meter().create_histogram("probe.scoped").record(1.0)

    span_scope = spans.get_finished_spans()[-1].instrumentation_scope
    metric_scope = _metric_scopes(reader)["probe.scoped"]
    for scope in (span_scope, metric_scope):
        assert (scope.name, scope.version, scope.schema_url) == (
            "redisvl",
            __version__,
            otel.SCHEMA_URL,
        )


def test_off_switch_silences_redisvl_despite_a_configured_sdk(otel_sdk, monkeypatch):
    spans, reader = otel_sdk
    monkeypatch.setenv(otel.DISABLED_ENV_VAR, "TRUE")

    with otel.get_tracer().start_as_current_span("probe") as span:
        span.set_attribute(otel.CACHE_NAME, "cache")
    otel.get_meter().create_histogram("probe.disabled").record(1.0)

    assert spans.get_finished_spans() == ()
    assert "probe.disabled" not in _metric_scopes(reader)


def test_without_an_sdk_redisvl_configures_nothing_and_costs_microseconds():
    """With the API alone, RedisVL must leave the global providers as the API's
    proxies, never initialise redis-py's observability, and record nothing.

    Runs in a fresh interpreter so no other test's SDK is configured. The
    instrumented path measured about 2 us on an Apple M4 Pro; the bound only
    catches gross regressions on loaded CI runners.
    """
    pytest.importorskip("opentelemetry")
    code = textwrap.dedent("""
        import json, timeit
        from opentelemetry import metrics, trace
        from redisvl.utils import otel

        tracer = otel.get_tracer()
        histogram = otel.get_meter().create_histogram("probe.overhead")
        attributes = {otel.CACHE_NAME: "cache", otel.CACHE_RESULT: "hit"}

        def call():
            with tracer.start_as_current_span("probe", attributes=attributes) as span:
                histogram.record(0.001, attributes)
                histogram.record(0.05, attributes)
            return span

        try:  # redis-py 7.2+
            from redis.observability import providers
            redis_py_initialised = providers._global_provider_manager is not None
        except ImportError:
            redis_py_initialised = False

        print(json.dumps({
            "recording": call().is_recording(),
            "tracer_provider": type(trace.get_tracer_provider()).__name__,
            "meter_provider": type(metrics.get_meter_provider()).__name__,
            "redis_py_initialised": redis_py_initialised,
            "seconds_per_call": min(timeit.repeat(call, number=10_000, repeat=5))
            / 10_000,
        }))
        """)
    result = json.loads(
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
    )

    assert result.pop("seconds_per_call") < 25e-6
    assert result == {
        "recording": False,
        "tracer_provider": "ProxyTracerProvider",
        "meter_provider": "_ProxyMeterProvider",
        "redis_py_initialised": False,
    }


def test_opentelemetry_is_only_ever_an_optional_dependency():
    requirements = [
        Requirement(line)
        for line in importlib.metadata.requires("redisvl") or []
        if Requirement(line).name.startswith("opentelemetry")
    ]
    assert requirements, "expected the otel extra to require opentelemetry-api"
    for requirement in requirements:
        assert requirement.marker is not None and not requirement.marker.evaluate(
            {"extra": ""}
        ), f"{requirement} is installed without an extra"


def test_without_opentelemetry_redisvl_imports_and_uses_its_stand_ins():
    """With OpenTelemetry hidden, no RedisVL module may fail to import because
    of it, and the stand-ins must work.

    Runs in a fresh interpreter, where an import hook makes every
    ``opentelemetry`` import fail as it would if the package were absent. A
    failure counts only when RedisVL's own code made the import: fastmcp, for
    one, requires OpenTelemetry itself, and other modules may lack their own
    optional dependencies.
    """
    code = textwrap.dedent("""
        import importlib, json, pkgutil, sys, traceback
        from importlib.abc import MetaPathFinder
        from pathlib import Path

        class HideOpenTelemetry(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "opentelemetry" or name.startswith("opentelemetry."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None

        sys.meta_path.insert(0, HideOpenTelemetry())

        import redisvl
        from redisvl.utils import otel

        package = Path(redisvl.__file__).parent

        def importer(error):
            # The innermost frame that is not the import machinery or this hook.
            frames = [
                frame.filename
                for frame in traceback.extract_tb(error.__traceback__)
                if not frame.filename.startswith("<")
            ]
            return Path(frames[-1]) if frames else None

        broken = {}
        for module in pkgutil.walk_packages(redisvl.__path__, "redisvl."):
            try:
                importlib.import_module(module.name)
            except ImportError as error:
                source = importer(error)
                if (error.name or "").startswith("opentelemetry") and (
                    source is not None and package in source.parents
                ):
                    broken[module.name] = f"{source.name}: {error}"

        with otel.get_tracer().start_as_current_span("probe") as span:
            span.set_attribute(otel.CACHE_NAME, "cache")
        otel.get_meter().create_histogram("probe").record(1.0)

        print(json.dumps({
            "broken": broken,
            "recording": span.is_recording(),
            "opentelemetry_loaded": any(
                name.startswith("opentelemetry") for name in sys.modules
            ),
        }))
        """)
    result = json.loads(
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
    )

    assert result == {"broken": {}, "recording": False, "opentelemetry_loaded": False}
