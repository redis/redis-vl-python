# LLM Router Extension - Design Document

## Overview

The LLM Router is an extension to RedisVL that provides intelligent, cost-optimized LLM model selection using semantic routing. Instead of routing queries to topics (like SemanticRouter), it routes queries to **model tiers** - selecting the cheapest LLM capable of handling each task.

## Problem Statement

### The LLM Cost Problem
Modern applications often default to using the most capable (and expensive) LLM for all queries, even when simpler models would suffice:
- "Hello, how are you?" -> Claude Opus 4.5 ($5/M tokens)
- "Hello, how are you?" -> GPT-4.1 Nano ($0.10/M tokens)

### Existing Solutions and Their Limitations

**RouteLLM** (CMU/LMSys):
- Binary classification only (strong vs weak model)
- No support for >2 tiers
- Requires training data or preference matrices

**NVIDIA LLM Router Blueprint**:
- Complexity classification approach (simple/moderate/complex)
- Provides the taxonomy basis but no open-source Redis-native implementation

**RouterArena / Bloom's Taxonomy Approach**:
- Maps query complexity to Bloom's cognitive levels
- Informs our tier design but lacks production routing infrastructure

**OpenRouter Auto-Router**:
- Black box routing decisions
- Data flows through third-party servers
- No transparency into why a model was selected
- Can't self-host or customize

**NotDiamond**:
- Proprietary ML model for routing
- Requires API calls for every routing decision
- No local/offline capability

**FrugalGPT**:
- Sequential cascade approach (try cheap first, escalate)
- Higher latency due to serial model calls

## Solution: Semantic Model Tier Routing

Repurpose RedisVL's battle-tested SemanticRouter for model selection:

```
SemanticRouter          ->    LLMRouter
-----------------------------------------
Route                   ->    ModelTier
route.name              ->    tier.name (simple/standard/expert)
route.references        ->    tier.references (task complexity examples)
route.metadata          ->    tier.metadata (cost, capabilities)
RouteMatch              ->    LLMRouteMatch (includes model string)
```

### Architecture

```
+---------------------------------------------------------------+
|                         LLMRouter                             |
+---------------------------------------------------------------+
|  +-------------+  +-------------+  +-------------+           |
|  |   Simple    |  |  Standard   |  |   Expert    |           |
|  |   Tier      |  |    Tier     |  |    Tier     |           |
|  +-------------+  +-------------+  +-------------+           |
|  | gpt-4.1-nano|  | sonnet 4.5  |  | opus 4.5    |           |
|  | $0.10/M     |  | $3/M        |  | $5/M        |           |
|  | threshold:  |  | threshold:  |  | threshold:  |           |
|  | 0.5         |  | 0.6         |  | 0.7         |           |
|  +-------------+  +-------------+  +-------------+           |
|         |                |                |                   |
|         +----------------+----------------+                   |
|                          v                                    |
|              +------------------------+                       |
|              |   Redis Vector Index   |                       |
|              |   (reference phrases)  |                       |
|              +------------------------+                       |
+---------------------------------------------------------------+
                           |
                           v
                    +-------------+
                    |   Query     |
                    |  "analyze   |
                    |   this..."  |
                    +-------------+
                           |
                           v
                    +-------------+
                    |  LiteLLM    |
                    |  (optional) |
                    +-------------+
```

## Key Design Decisions

### 1. Model Tiers, Not Individual Models

Routes map to **tiers** (simple, standard, expert) rather than specific models. This provides:
- Abstraction from model churn (swap haiku -> gemini-flash without changing routes)
- Clear mental model for users
- Easy cost optimization within tiers

### 2. Bloom's Taxonomy-Grounded Tiers

The default pretrained config maps tiers to Bloom's Taxonomy cognitive levels:
- **Simple** (Remember/Understand): Factual recall, greetings, format conversion
- **Standard** (Apply/Analyze): Code explanation, summarization, moderate analysis
- **Expert** (Evaluate/Create): Research, architecture, formal reasoning

This is informed by RouterArena's finding that cognitive complexity correlates with model capability requirements.

### 3. LiteLLM-Compatible Model Strings

Tier model identifiers use LiteLLM format (`provider/model`):
```python
ModelTier(
    name="standard",
    model="anthropic/claude-sonnet-4-5",  # Works directly with LiteLLM
    ...
)
```

### 4. Per-Tier Distance Thresholds

Each tier has its own `distance_threshold`, allowing fine-grained control:
```python
simple_tier = ModelTier(..., distance_threshold=0.5)   # Strict match
expert_tier = ModelTier(..., distance_threshold=0.7)   # Looser match
```

### 5. Cost-Aware Routing

When `cost_optimization=True`, the router adds a cost penalty to distances:
```python
adjusted_distance = distance + (cost_per_1k * cost_weight)
```
This prefers cheaper tiers when semantic distances are close.

### 6. Pretrained Configs with Embedded Vectors

The built-in `default.json` provides a ready-to-use 3-tier configuration:
```python
# Instant setup - no embedding model needed at load time
router = LLMRouter.from_pretrained("default", redis_client=client)
```

The pretrained config includes pre-computed embeddings from
`sentence-transformers/all-mpnet-base-v2`, with 18 reference phrases per tier
covering the Bloom's Taxonomy spectrum.

Custom configs can also be exported and shared:
```python
# Export (one-time, with embedding model)
router.export_with_embeddings("my_router.json")

# Import (no embedding needed)
router = LLMRouter.from_pretrained("my_router.json", redis_client=client)
```

### 7. Async Support

`AsyncLLMRouter` provides the same functionality using async I/O. Since
`__init__` cannot be async, it uses a `create()` classmethod factory:

```python
router = await AsyncLLMRouter.create(
    name="my-router",
    tiers=tiers,
    redis_client=async_client,
)
match = await router.route("hello")
```

Key async method mapping:

| Sync (`LLMRouter`) | Async (`AsyncLLMRouter`) |
|---------------------|--------------------------|
| `__init__()` | `await create()` |
| `from_existing()` | `await from_existing()` |
| `route()` | `await route()` |
| `route_many()` | `await route_many()` |
| `add_tier()` | `await add_tier()` |
| `remove_tier()` | `await remove_tier()` |
| `from_dict()` | `await from_dict()` |
| `from_pretrained()` | `await from_pretrained()` |
| `delete()` | `await delete()` |

## Module Structure

```
redisvl/extensions/llm_router/
+-- __init__.py              # Public exports (LLMRouter, AsyncLLMRouter, schemas)
+-- DESIGN.md                # This document
+-- schema.py                # Pydantic models
|   +-- ModelTier            # Tier definition
|   +-- LLMRouteMatch        # Routing result
|   +-- RoutingConfig        # Router configuration
|   +-- Pretrained*          # Export/import schemas
+-- router.py                # LLMRouter + AsyncLLMRouter implementations
+-- pretrained/
    +-- __init__.py          # Pretrained loader (get_pretrained_path)
    +-- default.json         # Standard 3-tier config (simple/standard/expert)
```

## API Examples

### Basic Usage

```python
from redisvl.extensions.llm_router import LLMRouter, ModelTier

tiers = [
    ModelTier(
        name="simple",
        model="openai/gpt-4.1-nano",
        references=[
            "hello", "hi there", "thanks", "goodbye",
            "what time is it?", "how are you?",
        ],
        metadata={"cost_per_1k_input": 0.0001},
        distance_threshold=0.5,
    ),
    ModelTier(
        name="standard",
        model="anthropic/claude-sonnet-4-5",
        references=[
            "analyze this code for bugs",
            "explain how neural networks learn",
            "compare and contrast these approaches",
        ],
        metadata={"cost_per_1k_input": 0.003},
        distance_threshold=0.6,
    ),
    ModelTier(
        name="expert",
        model="anthropic/claude-opus-4-5",
        references=[
            "prove this mathematical theorem",
            "architect a distributed system",
            "write a research paper analyzing",
        ],
        metadata={"cost_per_1k_input": 0.005},
        distance_threshold=0.7,
    ),
]

router = LLMRouter(
    name="my-llm-router",
    tiers=tiers,
    redis_url="redis://localhost:6379",
)

# Route a query
match = router.route("hello, how's it going?")
print(match.tier)   # "simple"
print(match.model)  # "openai/gpt-4.1-nano"

# Use with LiteLLM (optional integration)
from litellm import completion
response = completion(model=match.model, messages=[{"role": "user", "content": query}])
```

### Cost-Optimized Routing

```python
router = LLMRouter(
    name="cost-aware-router",
    tiers=tiers,
    cost_optimization=True,  # Prefer cheaper tiers when distances are close
    redis_url="redis://localhost:6379",
)
```

### Pretrained Router

```python
# Load without needing an embedding model for the references
router = LLMRouter.from_pretrained(
    "default",  # Built-in config, or path to JSON
    redis_client=client,
)
```

### Async Usage

```python
from redisvl.extensions.llm_router import AsyncLLMRouter

router = await AsyncLLMRouter.create(
    name="my-async-router",
    tiers=tiers,
    redis_url="redis://localhost:6379",
)

match = await router.route("explain how garbage collection works")
print(match.model)  # "anthropic/claude-sonnet-4-5"

# Or load from pretrained
router = await AsyncLLMRouter.from_pretrained("default", redis_client=client)

await router.delete()
```

## Comparison with SemanticRouter

| Feature | SemanticRouter | LLMRouter |
|---------|---------------|-----------|
| Purpose | Topic classification | Model selection |
| Output | Route name | Model string + metadata |
| Cost awareness | No | Yes |
| Pretrained configs | No | Yes |
| Per-route thresholds | Yes | Yes |
| LiteLLM integration | No | Yes (model strings) |
| Async support | No | Yes (`AsyncLLMRouter`) |

## Testing

### Unit Tests (`tests/unit/test_llm_router_schema.py`)
- Schema validation
- Pydantic model behavior
- Threshold bounds
- Empty/invalid inputs

### Integration Tests (`tests/integration/test_llm_router.py`)
- Router initialization
- Routing accuracy
- Cost optimization behavior
- Serialization (dict, YAML, JSON)
- Pretrained import/export
- Pretrained config loading (`from_pretrained("default")`)
- Tier management (add, remove, update)
- Persistence (from_existing)

### Async Integration Tests (`tests/integration/test_async_llm_router.py`)
- Mirrors all sync tests with `AsyncLLMRouter`
- Uses `async_client` fixture and async skip helpers
- Tests `create()` factory, async routing, serialization, tier management
- Pretrained config loading

Run tests:
```bash
uv run pytest tests/unit/test_llm_router_schema.py -v
uv run pytest tests/integration/test_llm_router.py -v
uv run pytest tests/integration/test_async_llm_router.py -v
```

## Future Enhancements

### 1. `complete()` Method
Direct LiteLLM integration for one-liner usage:
```python
response = router.complete("analyze this code", messages=[...])
```

### 2. Capability Filtering
Filter tiers by capability before routing:
```python
match = router.route("generate an image", capabilities=["vision"])
```

### 3. Budget Constraints
Enforce cost limits:
```python
router = LLMRouter(..., max_cost_per_1k=0.01)  # Never select opus
```

### 4. Fallback Chains
Define fallback order when primary tier unavailable:
```python
tier = ModelTier(..., fallback=["standard", "simple"])
```

## References

- [RedisVL SemanticRouter](https://docs.redisvl.com/en/latest/user_guide/semantic_router.html)
- [LiteLLM Model List](https://docs.litellm.ai/docs/providers)
- [RouteLLM](https://github.com/lm-sys/RouteLLM) - LMSys binary router framework
- [NVIDIA LLM Router Blueprint](https://build.nvidia.com/blueprints/llm-router) - Complexity-based routing
- [RouterArena / Bloom's Taxonomy](https://arxiv.org/abs/2412.06644) - Cognitive complexity for routing
- [FrugalGPT](https://arxiv.org/abs/2305.05176) - Cost-efficient LLM strategies
- [OpenRouter](https://openrouter.ai/) - Auto-routing concept
- [NotDiamond](https://notdiamond.ai/) - ML-based model routing
- [Unify.ai](https://unify.ai/) - Quality-cost tradeoff routing
