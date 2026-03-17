# Index Migrator Workspace

## Overview

This directory is the sole source of truth for RedisVL index migration planning.

No implementation should start unless the corresponding task exists in a `*_tasks.md` file in this directory.

This workspace is organized around two phases:

- Phase 1 MVP: `drop_recreate`
- Phase 2: `iterative_shadow`

The overall initiative covers both simple schema-only rebuilds and harder migrations that change vector dimensions, datatypes, precision, algorithms, or payload shape. Those advanced cases are intentionally delivered after the MVP rather than being treated as out of scope for the product.

The planning goal is to make handoff simple. Another engineer or process should be able to open this directory, read the active spec and task list, and start implementation without needing to rediscover product decisions.

## Guiding Principles

- Prefer simple and safe over clever orchestration.
- Reuse existing RedisVL primitives before adding new abstractions.
- Migrate one index at a time.
- Keep cutover and platform scaling operator-owned.
- Fail closed on unsupported schema changes.
- Treat documentation artifacts as implementation inputs, not as narrative background.

## Phase Status

| Phase | Mode | Status | Implementation Target |
| --- | --- | --- | --- |
| Phase 1 | `drop_recreate` | Ready | Yes |
| Phase 2 | `iterative_shadow` | Planned | No |

## Doc Map

- [01_context.md](./01_context.md): customer problem, constraints, and why the work is phased
- [02_architecture.md](./02_architecture.md): shared architecture, responsibilities, capacity model, and diagrams
- [03_benchmarking.md](./03_benchmarking.md): migration benchmarking goals, metrics, scenarios, and output artifacts
- [90_prd.md](./90_prd.md): final product requirements document for team review
- [10_v1_drop_recreate_spec.md](./10_v1_drop_recreate_spec.md): decision-complete MVP spec
- [11_v1_drop_recreate_tasks.md](./11_v1_drop_recreate_tasks.md): implementable MVP task list
- [12_v1_drop_recreate_tests.md](./12_v1_drop_recreate_tests.md): MVP test plan
- [20_v2_iterative_shadow_spec.md](./20_v2_iterative_shadow_spec.md): future iterative shadow spec
- [21_v2_iterative_shadow_tasks.md](./21_v2_iterative_shadow_tasks.md): future iterative shadow tasks
- [22_v2_iterative_shadow_tests.md](./22_v2_iterative_shadow_tests.md): future iterative shadow test plan

## Current Truth

The active implementation target is Phase 1.

- Spec: [10_v1_drop_recreate_spec.md](./10_v1_drop_recreate_spec.md)
- Tasks: [11_v1_drop_recreate_tasks.md](./11_v1_drop_recreate_tasks.md)
- Tests: [12_v1_drop_recreate_tests.md](./12_v1_drop_recreate_tests.md)

## Next Actions

- `V1-T01`
- `V1-T02`
- `V1-T03`

## Locked Decisions

- The planning workspace lives entirely under `nitin_docs/index_migrator/`.
- The top-level migration notes have been removed to avoid competing sources of truth.
- Phase 1 is documentation-backed implementation scope.
- Phase 2 stays planned until Phase 1 is implemented and learnings are folded back into this directory.
- The default artifact format for plans and reports is YAML.
- Benchmarking is required for migration duration, query impact, and resource-impact planning, but it should be implemented with simple structured outputs rather than a separate benchmarking framework.
- The default execution unit is a single index.
- The default operational model is operator-owned downtime, cutover, and scaling.
- Phase 2 owns advanced vector and payload-shape migrations, including datatype, precision, dimension, and algorithm changes.
