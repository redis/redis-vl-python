#!/usr/bin/env python3
"""Verify migration actually produces correct float16 conversions of original float32 data."""
import shutil
import tempfile
import time

import numpy as np
import redis

DIMS = 256
N = 1000
PREFIX = "verify_test:"

r = redis.from_url("redis://localhost:6379")

# 1. Create known vectors
print(f"Creating {N} docs with known float32 vectors ({DIMS}d)...")
original_vectors = {}
pipe = r.pipeline(transaction=False)
for i in range(N):
    vec = np.random.randn(DIMS).astype(np.float32)
    original_vectors[i] = vec.copy()
    pipe.hset(f"{PREFIX}{i}", mapping={"text": f"doc {i}", "embedding": vec.tobytes()})
pipe.execute()

# 2. Create HNSW index
r.execute_command(
    "FT.CREATE", "verify_idx", "ON", "HASH", "PREFIX", "1", PREFIX,
    "SCHEMA", "text", "TEXT",
    "embedding", "VECTOR", "HNSW", "10",
    "TYPE", "FLOAT32", "DIM", str(DIMS),
    "DISTANCE_METRIC", "COSINE", "M", "16", "EF_CONSTRUCTION", "200",
)
time.sleep(3)

# 3. Verify float32 stored correctly
pipe = r.pipeline(transaction=False)
for i in range(N):
    pipe.hget(f"{PREFIX}{i}", "embedding")
pre = pipe.execute()
f32_ok = all(np.array_equal(np.frombuffer(pre[i], dtype=np.float32), original_vectors[i]) for i in range(N))
print(f"Float32 pre-migration: {'PASS' if f32_ok else 'FAIL'}")

# 4. Run migration
print("\nRunning migration: HNSW float32 -> FLAT float16...")
import os
import yaml
from redisvl.migration.planner import MigrationPlanner
from redisvl.migration.executor import MigrationExecutor

backup_dir = tempfile.mkdtemp()
schema_patch = {
    "version": 1,
    "changes": {
        "update_fields": [{
            "name": "embedding",
            "attrs": {"algorithm": "flat", "datatype": "float16", "distance_metric": "cosine"},
        }],
    },
}
patch_path = os.path.join(backup_dir, "patch.yaml")
with open(patch_path, "w") as f:
    yaml.dump(schema_patch, f)

plan = MigrationPlanner().create_plan(
    index_name="verify_idx", redis_url="redis://localhost:6379",
    schema_patch_path=patch_path,
)
report = MigrationExecutor().apply(
    plan, redis_url="redis://localhost:6379",
    backup_dir=backup_dir, batch_size=200, num_workers=1,
)
print(f"  Result: {report.result}")
print(f"  Doc count: {report.validation.doc_count_match}")
print(f"  Schema: {report.validation.schema_match}")

# 5. THE REAL CHECK
print("\n=== DATA CORRECTNESS CHECK ===")
pipe = r.pipeline(transaction=False)
for i in range(N):
    pipe.hget(f"{PREFIX}{i}", "embedding")
post = pipe.execute()

missing = wrong_size = value_errors = 0
max_abs = 0.0
total_abs = 0.0
max_rel = 0.0

for i in range(N):
    data = post[i]
    if data is None:
        missing += 1
        continue
    if len(data) != DIMS * 2:
        wrong_size += 1
        continue

    actual_f16 = np.frombuffer(data, dtype=np.float16)
    expected_f16 = original_vectors[i].astype(np.float16)

    if not np.array_equal(actual_f16, expected_f16):
        value_errors += 1
        if value_errors <= 3:
            diff = np.abs(actual_f16.astype(np.float32) - expected_f16.astype(np.float32))
            print(f"  doc {i}: max_diff={diff.max():.8f}")
            print(f"    expected[:5] = {expected_f16[:5]}")
            print(f"    actual[:5]   = {actual_f16[:5]}")

    abs_err = np.abs(actual_f16.astype(np.float32) - original_vectors[i])
    max_abs = max(max_abs, abs_err.max())
    total_abs += abs_err.mean()

    nz = np.abs(original_vectors[i]) > 1e-10
    if nz.any():
        rel = abs_err[nz] / np.abs(original_vectors[i][nz])
        max_rel = max(max_rel, rel.max())

print(f"\nMissing docs:       {missing}")
print(f"Wrong size:         {wrong_size}")
print(f"Value mismatches:   {value_errors} (actual != expected float16)")
print(f"Max abs error:      {max_abs:.8f} (vs original float32)")
print(f"Avg abs error:      {total_abs/N:.8f}")
print(f"Max relative error: {max_rel:.6f} ({max_rel*100:.4f}%)")

if missing == 0 and wrong_size == 0 and value_errors == 0:
    print("\n✅ ALL DATA CORRECT: every vector is the exact float16 conversion of its original float32")
else:
    print(f"\n❌ ISSUES FOUND")

# Cleanup
try:
    r.execute_command("FT.DROPINDEX", "verify_idx")
except Exception:
    pass
pipe = r.pipeline(transaction=False)
for i in range(N):
    pipe.delete(f"{PREFIX}{i}")
pipe.execute()
shutil.rmtree(backup_dir, ignore_errors=True)
r.close()
