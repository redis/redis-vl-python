# LLM Router Extension - Design Document

## Overview

The LLM Router is an extension to RedisVL that provides intelligent, cost-optimized LLM model selection using semantic routing. Instead of routing queries to topics (like SemanticRouter), it routes queries to **model tiers** - selecting the cheapest LLM capable of handling each task.

## Problem Statement

### The LLM Cost Problem
Modern applications often default to using the most capable (and expensive) LLM for all queries, even when simpler models would suffice:
- "Hello, how are you?" → Claude Opus ($15/M tokens) ❌
- "Hello, how are you?" → Claude Haiku ($0.25/M tokens) ✅

### Existing Solutions and Their Limitations

**OpenRouter Auto-Router**:
- Black box routing decisions
- Data flows through third-party servers
- No transparency into why a model was selected
- Can't self-host or customize

**NotDiamond**:
- Proprietary ML model for routing
- Requires API calls for every routing decision
- No local/offline capability

**Manual Selection**:
- Developers hardcode model choices
- No dynamic adaptation to query complexity
- Maintenance burden as models evolve

## Solution: Semantic Model Tier Routing

Repurpose RedisVL's battle-tested SemanticRouter for model selection:

```
SemanticRouter          →    LLMRouter
─────────────────────────────────────────
Route                   →    ModelTier
route.name              →    tier.name (simple/reasoning/expert)
route.references        →    tier.references (task complexity examples)
route.metadata          →    tier.metadata (cost, capabilities)
RouteMatch              →    LLMRouteMatch (includes model string)
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LLMRouter                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Simple    │  │  Reasoning  │  │   Expert    │             │
│  │   Tier      │  │    Tier     │  │    Tier     │             │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤             │
│  │ haiku       │  │ sonnet      │  │ opus        │             │
│  │ $0.25/M     │  │ $3/M        │  │ $15/M       │             │
│  │ threshold:  │  │ threshold:  │  │ threshold:  │             │
│  │ 0.5         │  │ 0.6         │  │ 0.7         │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   Redis Vector Index  │                          │
│              │   (reference phrases) │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Query     │
                    │  "analyze   │
                    │   this..."  │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  LiteLLM    │
                    │  (optional) │
                    └─────────────┘
```

## Key Design Decisions

### 1. Model Tiers, Not Individual Models

Routes map to **tiers** (simple, reasoning, expert) rather than specific models. This provides:
- Abstraction from model churn (swap haiku→gemini-flash without changing routes)
- Clear mental model for users
- Easy cost optimization within tiers

### 2. LiteLLM-Compatible Model Strings

Tier model identifiers use LiteLLM format (`provider/model`):
```python
ModelTier(
    name="reasoning",
    model="anthropic/claude-sonnet-4-5",  # Works directly with LiteLLM
    ...
)
```

### 3. Per-Tier Distance Thresholds

Each tier has its own `distance_threshold`, allowing fine-grained control:
```python
simple_tier = ModelTier(..., distance_threshold=0.5)   # Strict match
expert_tier = ModelTier(..., distance_threshold=0.7)   # Looser match
```

### 4. Cost-Aware Routing

When `cost_optimization=True`, the router adds a cost penalty to distances:
```python
adjusted_distance = distance + (cost_per_1k * cost_weight)
```
This prefers cheaper tiers when semantic distances are close.

### 5. Pretrained Configs with Embedded Vectors

Export routers with pre-computed embeddings for instant setup:
```python
# Export (one-time, with embedding model)
router.export_with_embeddings("my_router.json")

# Import (no embedding needed)
router = LLMRouter.from_pretrained("my_router.json", redis_client=client)
```

This enables:
- Sharing router configs without requiring embedding models
- Faster initialization in production
- Consistent routing across deployments

## Module Structure

```
redisvl/extensions/llm_router/
├── __init__.py              # Public exports
├── DESIGN.md                # This document
├── schema.py                # Pydantic models
│   ├── ModelTier            # Tier definition
│   ├── LLMRouteMatch        # Routing result
│   ├── RoutingConfig        # Router configuration
│   └── Pretrained*          # Export/import schemas
├── router.py                # LLMRouter implementation
└── pretrained/
    ├── __init__.py          # Pretrained loader
    └── default.json         # Standard 3-tier config (TODO)
```

## API Examples

### Basic Usage

```python
from redisvl.extensions.llm_router import LLMRouter, ModelTier

tiers = [
    ModelTier(
        name="simple",
        model="anthropic/claude-haiku-4-5",
        references=[
            "hello", "hi there", "thanks", "goodbye",
            "what time is it?", "how are you?",
        ],
        metadata={"cost_per_1k_input": 0.00025},
        distance_threshold=0.5,
    ),
    ModelTier(
        name="reasoning",
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
        metadata={"cost_per_1k_input": 0.015},
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
print(match.model)  # "anthropic/claude-haiku-4-5"

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
# Load without needing an embedding model
router = LLMRouter.from_pretrained(
    "default",  # Built-in config, or path to JSON
    redis_client=client,
)
```

## Comparison with SemanticRouter

| Feature | SemanticRouter | LLMRouter |
|---------|---------------|-----------|
| Purpose | Topic classification | Model selection |
| Output | Route name | Model string + metadata |
| Cost awareness | ❌ | ✅ |
| Pretrained configs | ❌ | ✅ |
| Per-route thresholds | ✅ | ✅ |
| LiteLLM integration | ❌ | ✅ (model strings) |

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
- Tier management (add, remove, update)
- Persistence (from_existing)

Run tests:
```bash
uv run pytest tests/unit/test_llm_router_schema.py -v
uv run pytest tests/integration/test_llm_router.py -v
```

## Future Enhancements

### 1. `complete()` Method
Direct LiteLLM integration for one-liner usage:
```python
response = router.complete("analyze this code", messages=[...])
```

### 2. Async Support
```python
match = await router.aroute("query")
```

### 3. Capability Filtering
Filter tiers by capability before routing:
```python
match = router.route("generate an image", capabilities=["vision"])
```

### 4. Budget Constraints
Enforce cost limits:
```python
router = LLMRouter(..., max_cost_per_1k=0.01)  # Never select opus
```

### 5. Fallback Chains
Define fallback order when primary tier unavailable:
```python
tier = ModelTier(..., fallback=["reasoning", "simple"])
```

## References

- [RedisVL SemanticRouter](https://docs.redisvl.com/en/latest/user_guide/semantic_router.html)
- [LiteLLM Model List](https://docs.litellm.ai/docs/providers)
- [OpenRouter](https://openrouter.ai/) - Inspiration for auto-routing concept
- [NotDiamond](https://notdiamond.ai/) - ML-based model routing
