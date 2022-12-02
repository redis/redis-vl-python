from pathlib import Path

from redisvl.readers import PandasReader

data_location = str(Path(__file__).resolve().parent)

# def test_pandas_parquet():
#    reader = PandasReader.from_parquet("./data/simple_pandas.parquet")


def test_pandas_json(df):
    reader = PandasReader.from_json(data_location + "/data/simple_pandas.json")
    assert df.to_dict("records") == reader.records


def test_pandas_pickle(df):
    reader = PandasReader.from_pickle(data_location + "/data/simple_pandas.pkl")
    assert df.to_dict("records") == reader.records


def test_pandas_df(df):
    reader = PandasReader(df)
    assert df.to_dict("records") == reader.records
