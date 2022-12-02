import pickle
import typing as t

import pandas as pd

"""
Pandas Reader

Usage:
from redisvl.readers import PandasReader
reader = Pandas.from_pickle("tests/data/pandas.pickle")

"""


class PandasReader:
    def __init__(self, df: pd.DataFrame):
        self.records = df.to_dict("records")

    def __iter__(self):
        for record in self.records:
            yield record

    @classmethod
    def from_pickle(cls, data_file: str) -> t.List[dict]:
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
        return cls(df)

    @classmethod
    def from_json(cls, data_file: str) -> t.List[dict]:
        with open(data_file, "r") as f:
            df = pd.read_json(f)
        return cls(df)

    @classmethod
    def from_parquet(cls, data_file: str) -> t.List[dict]:
        try:
            import pyarrow
        except ImportError:
            # TODO make a better exception
            raise Exception("Pyarrow must be installed to use parquet functions")
        with open(data_file, "r") as f:
            df = pd.read_parquet(f)
        return cls(df)
