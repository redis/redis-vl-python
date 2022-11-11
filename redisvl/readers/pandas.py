import pickle
import typing as t

import pandas as pd


def from_pickle(data_file: str) -> t.List[dict]:
    """
    Read dataset from a pickled dataframe (Pandas) file.

    Args:
        data_file (str): Path to the destination
                         of the input data file.

    Returns:
        t.List[dict]: List of Hash objects to insert to Redis.
    """
    with open(data_file, "rb") as f:
        df = pickle.load(f)
    return df.to_dict("records")
