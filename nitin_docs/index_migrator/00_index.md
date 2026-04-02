# Index Migrator Workspace

## Overview

This directory contains the planning, design, and tracking documents for the RedisVL index migration feature.

Phase 1 (`drop_recreate`) has been implemented and shipped across a 6-PR stack. The implementation went beyond the original MVP spec to include vector quantization, field/prefix/index renames, async execution, batch operations, crash-safe reliability, and disk space estimation.

Phase 2 (`iterative_shadow`) remains planned and has not been started.

This workspace is preserved as a historical planning record and as the foundation for Phase 2 design.

## Guiding Principles

- Prefer simple and safe over clever orchestration.
- Reuse existing RedisVL primitives before adding new abstractions.
- Migrate one index at a time (batch mode migrates sequentially).
- Keep cutover and platform scaling operator-owned.
- Fail closed on unsupported schema changes.

## Phase Status

| Phase | Mode | Status | Notes |
| --- | --- | --- | --- |
| Phase 1 | `drop_recreate` | **Done** | Shipped as PRs #567-#572 |
| Phase 1+ | Extensions (async, batch, reliability) | **Done** | Shipped alongside Phase 1 |
| Phase 2 | `iterative_shadow` | Planned | Not started |

## Doc Map

### Planning (pre-implementation)
- [01_context.md](./01_context.md): customer problem, constraints, and why the work is phased
- [02_architecture.md](./02_architecture.md): shared architecture, responsibilities, capacity model, and diagrams
- [03_benchmarking.md](./03_benchmarking.md): migration benchmarking goals, metrics, scenarios, and output artifacts
- [90_prd.md](./90_prd.md): final product requirements document for team review

### Phase 1 (implemented)
- [04_implementation_summary.md](./04_implementation_summary.md): what was actually built, actual modules, actual CLI surface
- [05_migration_benchmark_report.md](./05_migration_benchmark_report.md): benchmark results (1K/10K/100K docs)
- [10_v1_drop_recreate_spec.md](./10_v1_drop_recreate_spec.md): original MVP spec (updated with implementation notes)
- [11_v1_drop_recreate_tasks.md](./11_v1_drop_recreate_tasks.md): task list (all completed)
- [12_v1_drop_recreate_tests.md](./12_v1_drop_recreate_tests.md): test plan

### Phase 2 (planned, not started)
- [20_v2_iterative_shadow_spec.md](./20_v2_iterative_shadow_spec.md): future iterative shadow spec
- [21_v2_iterative_shadow_tasks.md](./21_v2_iterative_shadow_tasks.md): future iterative shadow tasks
- [22_v2_iterative_shadow_tests.md](./22_v2_iterative_shadow_tests.md): future iterative shadow test plan

### Tracking
- [99_tickets.md](./99_tickets.md): all IM tickets with statuses
- [pr_comments.md](./pr_comments.md): collected PR review feedback

## Current Truth

Phase 1 is complete. The implementation source of truth is the code in `redisvl/migration/`.

For Phase 2 planning:
- Spec: [20_v2_iterative_shadow_spec.md](./20_v2_iterative_shadow_spec.md)
- Tasks: [21_v2_iterative_shadow_tasks.md](./21_v2_iterative_shadow_tasks.md)
- Tests: [22_v2_iterative_shadow_tests.md](./22_v2_iterative_shadow_tests.md)

## Next Actions

- Phase 1 implementation is complete. No remaining Phase 1 tasks.
- Phase 2 design review should begin once Phase 1 learnings are documented.
- See [99_tickets.md](./99_tickets.md) for backlog items (IM-B1, IM-11 through IM-20).

## Locked Decisions

- The planning workspace lives entirely under `nitin_docs/index_migrator/`.
- The default artifact format for plans and reports is YAML.
- Benchmarking is built into migration reporting, not a separate subsystem.
- The default execution unit is a single index (batch mode runs indexes sequentially).
- The default operational model is operator-owned downtime, cutover, and scaling.
- Phase 2 owns shadow migrations for incompatible changes that require running old and new indexes in parallel.
- Vector quantization, field renames, prefix changes, and index renames were added to Phase 1 scope during implementation (originally planned for Phase 2).
