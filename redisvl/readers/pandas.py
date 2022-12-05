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
    """The PandasReader class is used to read in data from a pandas dataframe
    that has been serialized to disk using pickle, json, parquet, or passed
    in memory. The data is then converted to a list of dictionaries that
    is passed to the SearchIndex class to be loaded into redis.
    """

    def __init__(self, df: pd.DataFrame):
        self.records = df.to_dict("records")

    def _convert_to_bytes(self, column):
        pass

    def __iter__(self):
        """Iterate over the records"""
        for record in self.records:
            yield record

    @classmethod
    def from_pickle(cls, data_file: str) -> t.List[dict]:
        """Read dataset from a pickled dataframe (Pandas).

        Args:
            data_file (str): Path to the destination
                            of the input data file.

        Returns:
            PandasReader: PandasReader object
        """
        with open(data_file, "rb") as f:
            df = pickle.load(f)
        return cls(df)

    @classmethod
    def from_json(cls, data_file: str) -> t.List[dict]:
        """Read dataset from a json file (df.to_json)

        Args:
            data_file (str): Path to the destination of the input data file.

        Returns:
            PandasReader: PandasReader object
        """
        with open(data_file, "r") as f:
            df = pd.read_json(f)
        return cls(df)

    @classmethod
    def from_parquet(cls, data_file: str) -> t.List[dict]:
        """Read dataset from a parquet file (df.to_parquet)

        Args:
            data_file (str): Path to the destination of the input data file.

        Returns:
            PandasReader: PandasReader object
        """
        try:
            import pyarrow
        except ImportError:
            # TODO make a better exception
            raise Exception("Pyarrow must be installed to use parquet functions")
        with open(data_file, "r") as f:
            df = pd.read_parquet(f)
        return cls(df)
