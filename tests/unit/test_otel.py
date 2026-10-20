"""RedisVL's OpenTelemetry plumbing: scope, off switch and the no-SDK path."""

import json
import subprocess
import sys
import textwrap

import pytest

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


@pytest.mark.parametrize(
    "value, enabled",
    [("SPAN_ONLY", True), ("span_and_event", True), ("true", False), ("", False)],
)
def test_only_the_span_modes_turn_on_content_capture(monkeypatch, value, enabled):
    # "true" turns capture on in older GenAI instrumentations, but not here.
    monkeypatch.setenv(otel.CAPTURE_CONTENT_ENV_VAR, value)
    otel.capture_content.cache_clear()
    try:
        assert otel.capture_content() is enabled
    finally:
        otel.capture_content.cache_clear()


def test_error_type_qualifies_all_but_builtin_exceptions():
    from redis.exceptions import ConnectionError as RedisConnectionError

    assert otel.error_type(ConnectionError()) == "ConnectionError"
    assert otel.error_type(RedisConnectionError()) == (
        "redis.exceptions.ConnectionError"
    )


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
