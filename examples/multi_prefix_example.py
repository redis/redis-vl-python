"""
Example: Multi-prefix index workaround

This script demonstrates how to:
1. Manually create a multi-prefix index using execute_command
2. Connect to it using SearchIndex.from_existing()
3. Load data with different prefixes using the keys parameter
4. Query and verify results come from both prefixes
"""

import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.redis.utils import array_to_buffer

# Connect to Redis
client = redis.Redis(host="localhost", port=6379, decode_responses=True)

INDEX_NAME = "user_simple"

# Clean up any existing index
try:
    client.ft(INDEX_NAME).dropindex(delete_documents=True)
except Exception:
    pass

# 1. Manually create the multi-prefix index
print("Creating multi-prefix index...")
client.execute_command(
    "FT.CREATE", INDEX_NAME,
    "ON", "HASH",
    "PREFIX", "2", "prefix_1:", "prefix_2:",
    "SCHEMA",
    "user", "TAG",
    "credit_score", "TAG",
    "job", "TEXT",
    "age", "NUMERIC",
    "user_embedding", "VECTOR", "FLAT", "6",
        "TYPE", "FLOAT32",
        "DIM", "3",
        "DISTANCE_METRIC", "COSINE"
)
print("Index created with prefixes: prefix_1:, prefix_2:")

# 2. Connect using from_existing
index = SearchIndex.from_existing(INDEX_NAME, redis_client=client)
print(f"Connected to index. Prefixes: {index.schema.index.prefix}")

# 3. Prepare test data
data_prefix_1 = [
    {
        "user": "john",
        "credit_score": "high",
        "job": "engineer",
        "age": 30,
        "user_embedding": array_to_buffer([0.1, 0.2, 0.3], dtype="float32"),
    },
    {
        "user": "jane",
        "credit_score": "medium",
        "job": "doctor",
        "age": 35,
        "user_embedding": array_to_buffer([0.2, 0.3, 0.4], dtype="float32"),
    },
]

data_prefix_2 = [
    {
        "user": "bob",
        "credit_score": "low",
        "job": "teacher",
        "age": 40,
        "user_embedding": array_to_buffer([0.3, 0.4, 0.5], dtype="float32"),
    },
    {
        "user": "alice",
        "credit_score": "high",
        "job": "lawyer",
        "age": 45,
        "user_embedding": array_to_buffer([0.4, 0.5, 0.6], dtype="float32"),
    },
]

# 4. Load data with explicit keys for each prefix
print("\nLoading data with prefix_1...")
keys_p1 = ["prefix_1:doc1", "prefix_1:doc2"]
index.load(data_prefix_1, keys=keys_p1)
print(f"Loaded keys: {keys_p1}")

print("\nLoading data with prefix_2...")
keys_p2 = ["prefix_2:doc1", "prefix_2:doc2"]
index.load(data_prefix_2, keys=keys_p2)
print(f"Loaded keys: {keys_p2}")

# 5. Query and verify we get results from both prefixes
print("\n" + "="*50)
print("Running vector search...")
print("="*50)

query = VectorQuery(
    vector=[0.2, 0.3, 0.4],
    vector_field_name="user_embedding",
    return_fields=["user", "credit_score", "job", "age"],
    num_results=10,
)

results = index.query(query)

print(f"\nFound {len(results)} results:\n")

prefix_1_count = 0
prefix_2_count = 0

for r in results:
    key = r.get("id", "")
    if key.startswith("prefix_1:"):
        prefix_1_count += 1
    elif key.startswith("prefix_2:"):
        prefix_2_count += 1
    print(f"  Key: {key}")
    print(f"    User: {r.get('user')}, Job: {r.get('job')}, Age: {r.get('age')}")
    print()

# 6. Verify both prefixes are represented
print("="*50)
print("VERIFICATION")
print("="*50)
print(f"Results from prefix_1: {prefix_1_count}")
print(f"Results from prefix_2: {prefix_2_count}")

if prefix_1_count > 0 and prefix_2_count > 0:
    print("\n✅ SUCCESS: Query returned results from BOTH prefixes!")
else:
    print("\n❌ FAILED: Expected results from both prefixes")

# Cleanup
print("\nCleaning up...")
client.ft(INDEX_NAME).dropindex(delete_documents=True)
print("Done!")

