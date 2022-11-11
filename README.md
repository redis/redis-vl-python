# RedisVL

A CLI and Library to help with loading data into Redis specifically for
usage with RediSearch and Redis Vector Search capabilities

### Usage

```
usage: redisvl <command> [<args>]

Commands:
        load        Load vector data into redis
        index       Index manipulation (create, delete, etc.)
        query       Query an existing index

Redis Vector load CLI

positional arguments:
  command     Subcommand to run

optional arguments:
  -h, --help  show this help message and exit

```

For any of the above commands, you will need to have an index schema written
into a yaml file for the cli to read. The format of the schema is as follows

```yaml
index:
  name: sample # index name used for querying
  storage_type: hash
  key_field: "id" # column name to use for key in redis
  prefix: vector  # prefix used for all loaded docs

# all fields to create index with
# sub-items correspond to redis-py Field arguments
fields:
  tag:
    categories: # name of a tag field used for queries
      separator: "|"
    year: # name of a tag field used for queries
      separator: "|"
  vector:
    vector: # name of the vector field used for queries
      datatype: "float32"
      algorithm: "flat" # flat or HSNW
      dims: 768
      distance_metric: "cosine" # ip, L2, cosine
```

#### Example Usage

```bash
# load in a pickled dataframe with
redisvl load -s sample.yml -d embeddings.pkl
```

```bash
# load in a pickled dataframe to a specific address and port
redisvl load -s sample.yml -d embeddings.pkl -h 127.0.0.1 -p 6379
```

```bash
# load in a pickled dataframe to a specific
# address and port and with password
redisvl load -s sample.yml -d embeddings.pkl -h 127.0.0.1 -p 6379 -p supersecret
```

### Support

#### Supported Index Fields

  - ``geo``
  - ``tag``
  - ``numeric``
  - ``vector``
  - ``text``
#### Supported Data Types
 - Pandas DataFrame (pickled)
#### Supported Redis Data Types
 - Hash
 - JSON (soon)

### Install
Install the Python requirements listed in `requirements.txt`.

```bash
git clone https://github.com/RedisVentures/data-loader.git
cd redisvl
pip install .
```

### Creating Input Data
#### Pandas DataFrame

  more to come, see tests and sample-data for usage