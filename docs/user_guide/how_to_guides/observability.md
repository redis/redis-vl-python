---
myst:
  html_meta:
    "description lang=en": |
      How to export SemanticCache traces and metrics with OpenTelemetry, and use them to tune the distance threshold.
---

# Observe the Semantic Cache with OpenTelemetry

RedisVL can report what `SemanticCache` does through [OpenTelemetry](https://opentelemetry.io/): a trace for every lookup that separates embedding time from search time, a hit rate, and the distance of the nearest match on every lookup, hits and misses alike. That last signal is what you need to tune `distance_threshold`.

RedisVL depends only on the OpenTelemetry API. Your application owns the SDK and the exporters, and RedisVL emits nothing until you configure them. With no SDK configured, the instrumentation adds about 11 µs to a lookup that embeds, measured on an Apple M4 Pro.

See [Cache LLM Responses](../03_llmcache.ipynb) for how to use the cache itself.

## Install

Install RedisVL with the `otel` extra, plus the SDK and an exporter of your choice:

```bash
pip install "redisvl[otel]" "opentelemetry-sdk>=1.32" opentelemetry-exporter-otlp-proto-http
```

Use `opentelemetry-sdk` 1.32.0 or later. RedisVL's histograms carry bucket advice, and earlier SDK releases drop it on OTLP export (fixed in 1.31.0) or apply only the first histogram's advice (fixed in 1.32.0).

## Configure an SDK

Set the global tracer and meter providers once, at application start-up. RedisVL picks them up automatically, whether you create the cache before or after; there is nothing RedisVL-specific to call.

```python
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from redisvl.extensions.cache.llm import SemanticCache

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(tracer_provider)

metrics.set_meter_provider(
    MeterProvider(metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())])
)

cache = SemanticCache(name="llmcache", redis_url="redis://localhost:6379")
cache.store(prompt="What is the capital of France?", response="Paris")
cache.check(prompt="What's the capital of France?")
```

The OTLP exporters send to `http://localhost:4318` unless you set `OTEL_EXPORTER_OTLP_ENDPOINT`, so run an OpenTelemetry Collector there, or point them at a backend that accepts OTLP. With nothing listening, the exporters log connection errors and retry, which can delay shutdown, but your application keeps working.

All RedisVL telemetry uses the instrumentation scope `redisvl`, with RedisVL's version and the semantic-conventions schema URL it targets.

## What RedisVL emits

### Spans

Every `SemanticCache` operation produces a span tree. A lookup looks like this:

```
semantic_cache.check     INTERNAL
├── embeddings {model}   CLIENT
└── retrieval {index}    CLIENT
```

`store`, `update` and `drop` produce `semantic_cache.store`, `semantic_cache.update` and `semantic_cache.drop`, and async methods produce the same trees as their sync forms. There is an embeddings span only when the operation embeds a prompt: a `check` or `store` given a precomputed `vector` has none, and `update` and `drop` never embed. The embeddings span comes from the vectorizer, so `SemanticRouter`, `SemanticMessageHistory` and direct calls to `embed` and `embed_many` produce it too.

| Span | Attribute | Value |
|---|---|---|
| `semantic_cache.*` | `redisvl.cache.name` | The cache name |
| | `redisvl.cache.result` | `hit` or `miss`, on `check` only |
| `embeddings {model}` | `gen_ai.operation.name` | `embeddings` |
| | `gen_ai.provider.name` | For example `openai`, `azure.ai.openai`, `cohere`, `mistral_ai`, `gcp.vertex_ai`, `gcp.gemini` or `aws.bedrock`; other vectorizers report their type, such as `hf` |
| | `gen_ai.request.model` | The vectorizer's model |
| | `gen_ai.embeddings.dimension.count` | The vectorizer's dimensions, when known |
| | `redisvl.embeddings_cache.result` | `hit`, `miss`, or `partial` for a batch; only when the vectorizer has an `EmbeddingsCache` and the call doesn't pass `skip_cache=True` |
| `retrieval {index}` | `gen_ai.operation.name` | `retrieval` |
| | `gen_ai.data_source.id`, `db.collection.name` | The index name |
| | `gen_ai.retrieval.top_k` | `num_results` |
| | `db.system.name` | `redis` |
| | `db.operation.name` | `FT.SEARCH` |
| | `redisvl.cache.distance_threshold` | The threshold the lookup used |
| | `redisvl.cache.distance` | The nearest candidate's distance, when there was one |

A failed operation sets an error status and `error.type` on every RedisVL span it passes through. `error.type` is the exception's class, prefixed by its module unless it is a builtin, such as `redis.exceptions.ConnectionError`.

### Metrics

| Metric | Unit | Attributes | Recorded on |
|---|---|---|---|
| `redisvl.semantic_cache.operation.duration` | `s` | `redisvl.cache.name`, `redisvl.cache.operation`, `redisvl.cache.result` (on `check`), `error.type` (on failure) | Every `check`, `store`, `update` and `drop`, sync or async |
| `redisvl.semantic_cache.distance` | `1` | `redisvl.cache.name`, `redisvl.cache.result` | Every completed lookup that found at least one candidate |
| `gen_ai.client.operation.duration` | `s` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `error.type` (on failure) | Every embedding call that reaches the provider, but not an embeddings-cache hit. One `embed_many` call records one point, covering all its batches |

There is no counter: the duration histogram's count is the number of operations. Cache-operation buckets start at 100 µs, because a lookup against local Redis takes about 250 µs before embedding, while embedding-duration buckets follow the GenAI conventions' advice, from 10 ms. Distance buckets are 0.01 wide up to 0.1, then coarser up to 2, the maximum cosine distance. To change any of them, use an SDK [View](https://opentelemetry.io/docs/languages/python/instrumentation/#metrics).

Names in the `gen_ai.*` namespace follow the OpenTelemetry GenAI conventions, which are still in Development status and may change in a future release.

## Tune the distance threshold

The distance histogram, split by result, shows how close your lookups come to the threshold, which defaults to 0.1:

| Pattern | What it suggests | What to do |
|---|---|---|
| Misses bunched just above the threshold | Some may be near-duplicates the cache is refusing | Consider loosening the threshold to turn the genuine ones into hits |
| Hits bunched just below the threshold | These are the matches most likely to be wrong | Consider tightening the threshold, trading hit rate for accuracy |
| Misses spread far above the threshold | Genuinely new prompts | Nothing: no threshold would serve them |

The distance of misses, as a heatmap:

```promql
sum by (le) (rate(redisvl_semantic_cache_distance_bucket{redisvl_cache_result="miss"}[1h]))
```

The share of misses within 0.05 of a 0.1 threshold:

```promql
sum(rate(redisvl_semantic_cache_distance_bucket{redisvl_cache_result="miss", le="0.15"}[1h]))
/
sum(rate(redisvl_semantic_cache_distance_count{redisvl_cache_result="miss"}[1h]))
```

The `le` value must be one of the bucket boundaries: a value between them, such as `0.23`, returns nothing. The metric does not record which threshold a lookup used, so after you change it, compare windows from before and after rather than mixing them.

A distance cannot tell you whether a match was right. Before you move the threshold, look at the prompts behind those distances with content capture on, as described under [Privacy](#privacy-and-content-capture).

## Starter queries

These PromQL queries use the names the OpenTelemetry Prometheus exporter produces. Other pipelines can translate names differently, so check your backend's metric names first.

Hit rate, per cache:

```promql
sum by (redisvl_cache_name) (rate(redisvl_semantic_cache_operation_duration_seconds_count{redisvl_cache_operation="check", redisvl_cache_result="hit"}[5m]))
/
sum by (redisvl_cache_name) (rate(redisvl_semantic_cache_operation_duration_seconds_count{redisvl_cache_operation="check", redisvl_cache_result=~"hit|miss"}[5m]))
```

Failed checks carry `error.type` and no result, so they count towards neither side.

95th-percentile duration, per operation:

```promql
histogram_quantile(0.95, sum by (le, redisvl_cache_operation) (rate(redisvl_semantic_cache_operation_duration_seconds_bucket[5m])))
```

95th-percentile embedding latency, per provider and model:

```promql
histogram_quantile(0.95, sum by (le, gen_ai_provider_name, gen_ai_request_model) (rate(gen_ai_client_operation_duration_seconds_bucket{gen_ai_operation_name="embeddings"}[5m])))
```

## Add redis-py's own metrics

redis-py 7.4 and later ship their own OpenTelemetry metrics for connections and commands. RedisVL never initialises them, so turn them on yourself after setting the meter provider. Command latency is not in the default groups:

```python
from redis.observability import MetricGroup, OTelConfig, get_observability_instance

get_observability_instance().init(
    OTelConfig(
        metric_groups=[
            MetricGroup.CONNECTION_BASIC,
            MetricGroup.RESILIENCY,
            MetricGroup.COMMAND,
        ]
    )
)
```

redis-py emits metrics only, not spans, which is why RedisVL traces the search itself as the `retrieval` span.

If you also run a provider's own GenAI instrumentation, such as one for the OpenAI SDK, both can report the same embedding call: expect a nested client span, and a second set of `gen_ai.client.operation.duration` points under that library's instrumentation scope.

## Privacy and content capture

By default RedisVL records no prompt, response, metadata or filter values, in any span, event or metric. It never records exception messages on its spans either, because they can contain prompt text: a failure records only the exception's class. Cache names and model identifiers are exported, so keep tenant or user ids out of cache names.

To see what the cache matched, set:

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
```

With `SPAN_ONLY` or `SPAN_AND_EVENT`, the `retrieval` span also records:

- `gen_ai.retrieval.query.text`: the prompt, whenever the lookup was passed one, even alongside a vector.
- `gen_ai.retrieval.documents`: a JSON list of the candidates' entry ids and scores. Each `score` is the raw vector distance, so lower is closer, unlike the relevance score the GenAI conventions describe. Entry ids are unsalted SHA-256 hashes of the stored prompt and filters, so treat them as content too.

RedisVL never records response text, but the variable is process-wide: other GenAI instrumentations that read it may also start recording prompts and responses. Unlike some of them, RedisVL does not treat `true` as a span mode, and needs no stability opt-in. It reads the variable once, on the first cache lookup.

Prompts can contain personal data. Capture content in place rather than copying production traffic elsewhere, send it only to a trace backend approved to hold prompt data, sample to limit the volume, and turn capture off again when you have what you need, which takes a restart.

## Turn it off

To silence RedisVL's telemetry while keeping the rest of your application's, set:

```bash
export REDISVL_OTEL_DISABLED=true
```

Only `true` (case-insensitive) disables it. RedisVL reads the variable once, the first time it records telemetry, so set it before your application starts.
