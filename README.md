# RediSearch Data Loader
The purpose of this script is to assist in loading datasets to a RediSearch instance efficiently.

The project is brand new and will undergo improvements over time.

## Getting Started

### Requirements
Install the Python requirements listed in `requirements.txt`.

```bash
$ pip install -r requirements.txt
```

### Data
In order to run the script you need to have a dataset that contains your vectors and metadata.

>Currently, the data file must be a pickled pandas dataframe. Support for more data types will be included in future iterations.

### Schema
Along with the dataset, you must update the dataset schema for RediSearch in [`data/schema.py`](data/schema.py).

### Running
The `main.py` script provides an entrypoint with optional arguments to upload your dataset to a Redis server.

#### Usage
```
python main.py

  -h, --help          Show this help message and exit
  --host              Redis host
  -p, --port          Redis port
  -a, --password      Redis password
  -c , --concurrency  Amount of concurrency
  -d , --data         Path to data file
  --prefix            Key prefix for all hashes in the search index
  -v , --vector       Vector field name in df
  -i , --index        Index name
```

#### Defaults

| Argument        | Default |
| ----------- | ----------- |
| Host | `localhost` |
| Port  | `6379` |
| Password | "" |
| Concurrency | `50` |
| Data (Path) | `data/embeddings.pkl` |
| Prefix | `vector:` |
| Vector (Field Name) | `vector` |
| Index Name | `index` |


#### Examples

Load to a local (default) redis server with a custom index name and with concurrency = 100:
```bash
$ python main.py -d data/embeddings.pkl -i myIndex -c 100
```

Load to a cloud redis server with all other defaults:
```bash
$ python main.py -h {redis-host} -p {redis-port} -a {redis-password}
```