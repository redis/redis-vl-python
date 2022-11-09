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
  name: sample
  storage_type: hash
  prefix: vector

fields:
  tag:
    categories:
      separator: "|"
    year:
      separator: "|"
  vector:
    vector:
      datatype: "float32"
      algorithm: "flat"
      dims: 768
      distance_metric: "cosine"
```

#### Example Usage

```bash
# load in a pickled dataframe with a column named "vector"
redisvl load -s sample.yml -d embeddings.pkl -v vector
```

```bash
# load in a pickled dataframe with a column named "vector" to a specific address and port
redisvl load -s sample.yml -d embeddings.pkl -v vector -h 127.0.0.1 -p 6379
```

```bash
# load in a pickled dataframe with a column named "vector" to a specific
# address and port and with password
redisvl load -s sample.yml -d embeddings.pkl -v vector -h 127.0.0.1 -p 6379 -p supersecret
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
