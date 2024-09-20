# EXAMPLE: query_vector
# HIDE_START
import json
import warnings
import redis
import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import RangeQuery, VectorQuery
from redisvl.schema import IndexSchema
from sentence_transformers import SentenceTransformer


def embed_text(model, text):
    return np.array(model.encode(text)).astype(np.float32).tobytes()

r = redis.Redis(decode_responses=True)

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*clean_up_tokenization_spaces.*")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# create index
schema = IndexSchema.from_yaml('data/query_vector_idx.yaml')
index = SearchIndex(schema, r)
index.create(overwrite=True, drop=True)

# load data
with open("data/query_vector.json") as f:
    bicycles = json.load(f)
index.load(bicycles)
# HIDE_END

# STEP_START vector1
query = "Bike for small kids"
query_vector = embed_text(model, query)
print(query_vector[:10]) # >>> b'\x02=c=\x93\x0e\xe0=aC'

vquery = VectorQuery(
    vector=query_vector,
    vector_field_name="description_embeddings",
    num_results=3,
    return_score=True,
    return_fields=["description"]
)
res = index.query(vquery)
print(res) # >>> [{'id': 'bicycle:6b702e8b...', 'vector_distance': '0.399111807346', 'description': 'Kids want...
# REMOVE_START
assert len(res) == 3
# REMOVE_END
# STEP_END

# STEP_START vector2
vquery = RangeQuery(
    vector=query_vector,
    vector_field_name="description_embeddings",
    distance_threshold=0.5,
    return_score=True
).return_fields("description").dialect(2)
res = index.query(vquery)
print(res) # >>> [{'id': 'bicycle:6bcb1bb4...', 'vector_distance': '0.399111807346', 'description': 'Kids want...
# REMOVE_START
assert len(res) == 2
# REMOVE_END
# STEP_END

# REMOVE_START
# destroy index and data
index.delete(drop=True)
# REMOVE_END
