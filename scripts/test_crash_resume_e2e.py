#!/usr/bin/env python3
"""E2E crash-resume test: 10,000 docs, float32→float16, 4 simulated crashes.

Strategy:
  - Single-worker with backup_dir for deterministic checkpoint tracking
  - Monkey-patch pipeline_write_vectors to raise after N batches
  - 10,000 docs / batch_size=500 = 20 batches total
  - Crash at batches: 5 (25%), 10 (50%), 15 (75%), 18 (90%)
  - Each resume verifies partial progress, then continues
  - Final resume completes and verifies all 10,000 docs are float16
"""
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import redis
import yaml

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
INDEX_NAME = "e2e_crash_test_idx"
PREFIX = "e2e_crash:"
NUM_DOCS = 10_000
DIMS = 128
BATCH_SIZE = 500
TOTAL_BATCHES = NUM_DOCS // BATCH_SIZE  # 20

# Crash after these many TOTAL batches have been quantized
CRASH_AFTER_BATCHES = [3, 7, 11, 16, 19]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cleanup_index(r):
    try:
        r.execute_command("FT.DROPINDEX", INDEX_NAME)
    except Exception:
        pass
    keys = list(r.scan_iter(match=f"{PREFIX}*", count=1000))
    while keys:
        r.delete(*keys[:500])
        keys = keys[500:]
        if not keys:
            keys = list(r.scan_iter(match=f"{PREFIX}*", count=1000))


def create_index_and_load(r):
    log(f"Creating index '{INDEX_NAME}' with {NUM_DOCS:,} docs ({DIMS}-dim float32)...")
    r.execute_command(
        "FT.CREATE", INDEX_NAME, "ON", "HASH", "PREFIX", "1", PREFIX,
        "SCHEMA", "title", "TEXT",
        "embedding", "VECTOR", "FLAT", "6",
        "TYPE", "FLOAT32", "DIM", str(DIMS), "DISTANCE_METRIC", "COSINE",
    )
    pipe = r.pipeline(transaction=False)
    for i in range(NUM_DOCS):
        vec = np.random.randn(DIMS).astype(np.float32).tobytes()
        pipe.hset(f"{PREFIX}{i}", mapping={"title": f"Doc {i}", "embedding": vec})
        if (i + 1) % 500 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
    pipe.execute()
    # Wait for indexing
    for _ in range(60):
        info = r.execute_command("FT.INFO", INDEX_NAME)
        info_dict = dict(zip(info[::2], info[1::2]))
        num_indexed = int(info_dict.get(b"num_docs", info_dict.get("num_docs", 0)))
        if num_indexed >= NUM_DOCS:
            break
        time.sleep(0.5)
    log(f"Index ready: {num_indexed:,} docs indexed")
    return num_indexed


def verify_vectors(r, expected_bytes, label=""):
    """Count docs by vector size. Returns (correct_count, wrong_count)."""
    pipe = r.pipeline(transaction=False)
    for i in range(NUM_DOCS):
        pipe.hget(f"{PREFIX}{i}", "embedding")
    results = pipe.execute()
    correct = sum(1 for d in results if d and len(d) == expected_bytes)
    wrong = NUM_DOCS - correct
    if label:
        log(f"  {label}: {correct:,} correct ({expected_bytes}B), {wrong:,} other")
    return correct, wrong


def count_quantized_docs(r, float16_bytes=256, float32_bytes=512):
    """Count how many docs are already float16 vs float32."""
    pipe = r.pipeline(transaction=False)
    for i in range(NUM_DOCS):
        pipe.hget(f"{PREFIX}{i}", "embedding")
    results = pipe.execute()
    f16 = sum(1 for d in results if d and len(d) == float16_bytes)
    f32 = sum(1 for d in results if d and len(d) == float32_bytes)
    return f16, f32


def make_plan(backup_dir):
    from redisvl.migration.planner import MigrationPlanner
    schema_patch = {
        "version": 1,
        "changes": {
            "update_fields": [{
                "name": "embedding",
                "attrs": {"algorithm": "flat", "datatype": "float16", "distance_metric": "cosine"},
            }]
        },
    }
    patch_path = os.path.join(backup_dir, "schema_patch.yaml")
    with open(patch_path, "w") as f:
        yaml.dump(schema_patch, f)
    planner = MigrationPlanner()
    plan = planner.create_plan(index_name=INDEX_NAME, redis_url=REDIS_URL, schema_patch_path=patch_path)
    # Save plan for resume
    plan_path = os.path.join(backup_dir, "plan.yaml")
    with open(plan_path, "w") as f:
        yaml.dump(plan.model_dump(), f, sort_keys=False)
    return plan, plan_path


class SimulatedCrash(Exception):
    """Raised to simulate a process crash during quantization."""
    pass



def run_attempt(plan, backup_dir, crash_after=None, attempt_num=0):
    """Run apply(). If crash_after is set, crash after that many total quantize batches.

    Uses direct monkey-patching of the module attribute to ensure the
    executor's local `from ... import` picks up the patched version.
    """
    from redisvl.migration.executor import MigrationExecutor
    import redisvl.migration.quantize as quantize_mod

    original_write = quantize_mod.pipeline_write_vectors
    executor = MigrationExecutor()
    events = []

    def progress_cb(step, detail=None):
        msg = f"  [{step}] {detail}" if detail else f"  [{step}]"
        events.append(msg)
        log(msg)

    if crash_after is not None:
        # Read backup to see how many batches already done
        from redisvl.migration.backup import VectorBackup
        safe = INDEX_NAME.replace("/", "_").replace("\\", "_").replace(":", "_")
        bp = str(os.path.join(backup_dir, f"migration_backup_{safe}"))
        existing = VectorBackup.load(bp)
        already_done = existing.header.quantize_completed_batches if existing else 0
        new_batches_allowed = crash_after - already_done
        call_counter = [0]

        log(f"  [attempt {attempt_num}] Crash after {crash_after} total batches "
            f"({already_done} already done, {new_batches_allowed} new allowed)")

        def crashing_write(client, converted):
            call_counter[0] += 1
            if call_counter[0] > new_batches_allowed:
                raise SimulatedCrash(
                    f"💥 Simulated crash at write call {call_counter[0]} "
                    f"(allowed {new_batches_allowed})!"
                )
            return original_write(client, converted)

        # Monkey-patch at module level
        quantize_mod.pipeline_write_vectors = crashing_write
        try:
            report = executor.apply(
                plan, redis_url=REDIS_URL, progress_callback=progress_cb,
                backup_dir=backup_dir, batch_size=BATCH_SIZE,
                num_workers=1, keep_backup=True,
            )
        finally:
            quantize_mod.pipeline_write_vectors = original_write
        log(f"  [attempt {attempt_num}] Write calls made: {call_counter[0]}")
        return report, events
    else:
        log(f"  [attempt {attempt_num}] Final run — no crash limit")
        report = executor.apply(
            plan, redis_url=REDIS_URL, progress_callback=progress_cb,
            backup_dir=backup_dir, batch_size=BATCH_SIZE,
            num_workers=1, keep_backup=True,
        )
        return report, events


def inspect_backup(backup_dir):
    """Read backup header and report state."""
    from redisvl.migration.backup import VectorBackup
    safe = INDEX_NAME.replace("/", "_").replace("\\", "_").replace(":", "_")
    bp = str(os.path.join(backup_dir, f"migration_backup_{safe}"))
    backup = VectorBackup.load(bp)
    if backup:
        h = backup.header
        log(f"  Backup: phase={h.phase}, dump_batches={h.dump_completed_batches}, "
            f"quantize_batches={h.quantize_completed_batches}")
        return h
    else:
        log("  Backup: not found")
        return None


def main():
    log("=" * 70)
    log(f"CRASH-RESUME E2E: {NUM_DOCS:,} docs, {DIMS}d, float32→float16")
    log(f"  Batch size: {BATCH_SIZE}, Total batches: {TOTAL_BATCHES}")
    log(f"  Crash points: {CRASH_AFTER_BATCHES} batches")
    log("=" * 70)

    r = redis.from_url(REDIS_URL)
    cleanup_index(r)

    num_docs = create_index_and_load(r)
    assert num_docs >= NUM_DOCS, f"Only {num_docs} indexed!"

    correct, _ = verify_vectors(r, 512, "Pre-migration float32")
    assert correct == NUM_DOCS

    backup_dir = tempfile.mkdtemp(prefix="crash_resume_backup_")
    log(f"\nBackup dir: {backup_dir}")

    plan, plan_path = make_plan(backup_dir)
    log(f"Plan: mode={plan.mode}, changes detected: "
        f"{len(plan.requested_changes.get('changes', {}).get('update_fields', []))}")

    try:
        # ── CRASH 1-4: Simulate crashes during quantization ──
        for crash_num, crash_at in enumerate(CRASH_AFTER_BATCHES):
            log(f"\n{'─'*60}")
            log(f"CRASH {crash_num + 1}/{len(CRASH_AFTER_BATCHES)}: "
                f"Crashing after batch {crash_at}/{TOTAL_BATCHES} "
                f"({crash_at * BATCH_SIZE:,} docs)")
            log(f"{'─'*60}")

            report, events = run_attempt(
                plan, backup_dir, crash_after=crash_at, attempt_num=crash_num + 1
            )
            log(f"  Result: {report.result}")

            # Verify backup state
            header = inspect_backup(backup_dir)
            assert header is not None, "Backup should exist after crash!"
            assert header.quantize_completed_batches == crash_at, (
                f"Expected {crash_at} batches quantized, got {header.quantize_completed_batches}"
            )
            assert header.phase in ("active", "ready"), (
                f"Expected phase 'active' or 'ready', got '{header.phase}'"
            )

            # Verify partial progress: some docs should be float16
            f16, f32 = count_quantized_docs(r)
            expected_f16 = crash_at * BATCH_SIZE
            log(f"  Partial progress: {f16:,} float16, {f32:,} float32")
            assert f16 == expected_f16, (
                f"Expected {expected_f16} float16 docs, got {f16}"
            )
            assert f32 == NUM_DOCS - expected_f16, (
                f"Expected {NUM_DOCS - expected_f16} float32 docs, got {f32}"
            )
            log(f"  ✅ Crash {crash_num + 1} verified: {f16:,} quantized, "
                f"{f32:,} remaining")

        # ── FINAL RESUME: Complete the migration ──
        log(f"\n{'─'*60}")
        log(f"FINAL RESUME: Completing remaining "
            f"{TOTAL_BATCHES - CRASH_AFTER_BATCHES[-1]} batches")
        log(f"{'─'*60}")

        report, events = run_attempt(plan, backup_dir, crash_after=None, attempt_num=5)
        log(f"  Result: {report.result}")
        assert report.result == "succeeded", f"Final resume failed: {report.result}"

        # Verify ALL docs are float16
        correct, wrong = verify_vectors(r, 256, "Post-migration float16")
        assert correct == NUM_DOCS, f"Only {correct}/{NUM_DOCS} docs are float16!"
        assert wrong == 0

        log(f"\n✅ ALL {NUM_DOCS:,} docs verified as float16!")

        # Verify backup is completed
        header = inspect_backup(backup_dir)
        assert header is not None
        assert header.phase == "completed"
        assert header.quantize_completed_batches == TOTAL_BATCHES

        log(f"\n{'='*70}")
        log("RESULTS")
        log(f"{'='*70}")
        log(f"  {NUM_DOCS:,} docs migrated float32→float16")
        log(f"  Crashes simulated: {len(CRASH_AFTER_BATCHES)}")
        for i, cb in enumerate(CRASH_AFTER_BATCHES):
            log(f"    Crash {i+1}: after batch {cb}/{TOTAL_BATCHES} "
                f"({cb*BATCH_SIZE:,}/{NUM_DOCS:,} docs)")
        log(f"  Final resume completed remaining {TOTAL_BATCHES - CRASH_AFTER_BATCHES[-1]} batches")
        log(f"  All {NUM_DOCS:,} vectors verified ✅")
        log(f"{'='*70}")

    finally:
        cleanup_index(r)
        shutil.rmtree(backup_dir, ignore_errors=True)
        r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
