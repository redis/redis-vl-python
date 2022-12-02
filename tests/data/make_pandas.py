import pickle

import numpy as np
import pandas as pd

data = pd.DataFrame(
    {
        "users": ["john", "mary", "joe"],
        "age": [1, 2, 3],
        "job": ["engineer", "doctor", "dentist"],
        "credit_score": ["high", "low", "medium"],
        "user_embedding": [
            [0.1, 0.1, 0.5],
            [0.1, 0.1, 0.5],
            [0.9, 0.9, 0.1],
        ],
    }
)

# data.to_parquet("./simple_pandas.parquet", index=False)
data.to_json("./simple_pandas.json")

with open("./simple_pandas.pkl", "wb") as f:
    f.write(pickle.dumps(data))
