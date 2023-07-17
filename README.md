# RedisVL: Python Client Library for Redis as a Vector Database


[![Codecov](https://img.shields.io/codecov/c/github/RedisVentures/RedisVL/dev?label=Codecov&logo=codecov&token=E30WxqBeJJ)](https://codecov.io/gh/RedisVentures/RedisVL)
[![License](https://img.shields.io/badge/License-BSD-3--blue.svg)](https://opensource.org/licenses/mit/)


RedisVL provides a powerful Python client library for using Redis as a Vector Database. Leverage the speed and reliability of Redis along with vector-based semantic search capabilities to supercharge your application!

**Note:** This project is rapidly evolving, and the API may change frequently. Always refer to the most recent [documentation](https://redisvl.com/docs).
## 🚀 What is RedisVL?

Vector databases have become increasingly popular in recent years due to their ability to store and retrieve vectors efficiently. However, most vector databases are complex to use and require a lot of time and effort to set up. RedisVL aims to solve this problem by providing a simple and intuitive interface for using Redis as a vector database.

RedisVL provides a client library that enables you to harness the power of Redis as a vector database. This library simplifies the process of storing, retrieving, and performing semantic searches on vectors in Redis. It also provides a robust index management system that allows you to create, update, and delete indices with ease.


### Capabilities

RedisVL has a host of powerful features designed to streamline your vector database operations.

1. **Index Management**: RedisVL allows for indices to be created, updated, and deleted with ease. A schema for each index can be defined in yaml or directly in python code and used throughout the lifetime of the index.

2. **Vector Creation**: RedisVL integrates with OpenAI and other embedding providers to make the process of creating vectors straightforward.

3. **Vector Search**: RedisVL provides robust search capabilities that enable you to query vectors synchronously and asynchronously. Hybrid queries that utilize tag, geographic, numeric, and other filters like full-text search are also supported.

4. **Semantic Caching**: ``LLMCache`` is a semantic caching interface built directly into RedisVL. It allows for the caching of generated output from LLM models like GPT-3 and others. As semantic search is used to check the cache, a threshold can be set to determine if the cached result is relevant enough to be returned. If not, the model is called and the result is cached for future use. This can increase the QPS and reduce the cost of using LLM models.


## 😊 Quick Start

Please note that this library is still under heavy development, and while you can quickly try RedisVL and deploy it in a production environment, the API may be subject to change at any time.

`pip install redisvl`

## Example Usage

### Index Management

Indices can be defined through yaml specification that corresponds directly to the RediSearch field names and arguments in redis-py

```yaml
index:
  name: users
  storage_type: hash
  prefix: "user:"
  key_field: "id"

fields:
  # define tag fields
  tag:
  - name: users
  - name: job
  - name: credit_store
  # define numeric fields
  numeric:
  - name: age
  # define vector fields
  vector:
  - name: user_embedding
    algorithm: hnsw
    distance_metric: cosine
```

This would correspond to a dataset that looked something like

| users | age |     job    | credit_score |           user_embedding          |
|-------|-----|------------|--------------|-----------------------------------|
| john  |  1  |  engineer  |     high     | \x3f\x8c\xcc\x3f\x8c\xcc?@         |
| mary  |  2  |   doctor   |     low      | \x3f\x8c\xcc\x3f\x8c\xcc?@         |
|  joe  |  3  |  dentist   |    medium    | \x3f\xab\xcc?\xab\xcc?@         |


With the schema, the RedisVL library can be used to create, load vectors and perform vector searches
```python
from redisvl.index import SearchIndex
from redisvl.query import create_vector_query

# define and create the index
index = SearchIndex.from_yaml("./users_schema.yml"))
index.connect("redis://localhost:6379")
index.create()

index.load(pd.read_csv("./users.csv").to_records())

query = create_vector_query(
    ["users", "age", "job", "credit_score"],
    number_of_results=2,
    vector_field_name="user_embedding",
)

query_vector = np.array([0.1, 0.1, 0.5]).tobytes()
results = index.search(query, query_params={"vector": query_vector})

```

### Semantic cache

The ``LLMCache`` Interface in RedisVL can be used as follows.

```python
# init open ai client
import openai
openai.api_key = "sk-xxx"

from redisvl.llmcache.semantic import SemanticCache
cache = SemanticCache(redis_host="localhost", redis_port=6379, redis_password=None)

def ask_gpt3(question):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=question,
      max_tokens=100
    )
    return response.choices[0].text.strip()

def answer_question(question: str):
    results = cache.check(question)
    if results:
        return results[0]
    else:
        answer = ask_gpt3(question)
        cache.store(question, answer)
        return answer
```


