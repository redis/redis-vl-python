# **Objective**

Make sql-like commands available to be translated into Redis queries via redisvl to cut down on syntax overhead for engineers.

Ex:

```py
from redisvl.query import SQLQuery

sql_query = SQLQuery("""
  SELECT title, author, price 
  FROM my_book_index
  WHERE category = "scify"
"""
)

response = redis_index.query(sql_query)
```

This code would then produce the equivalent redis query to be executed against the database:

```py
FT.search my_book_index
  "@category:{scify}"
  LOAD 3 @title @author @price
  DIALECT 2
```

# **Scope**

### **Disclaimers:**

* Redis is a nosql database therefore this conversion will **not allow** for SQL-like joins and other SQL specific querying patterns due to core data modeling differences.  
* Helper classes will be for **query-side** only \- meaning it will **not** be in scope to create or modify indexes via a SQL syntax.  
* We will also limit initial scope to target **specific SQL dialect**.   
  * Target dialect: \<input\>

### **In scope query examples:**

| Goal Functionality | Redis Example | SQL equivalent |
| :---- | :---- | :---- |
| Group by count query. |  FT.AGGREGATE KM\_2345 @created\_time:\[0 \+inf\] GROUPBY 7 @file\_id @file\_name @doc\_base\_id @created\_by @created\_time @last\_updated\_by @last\_updated\_time REDUCE COUNT 0 AS count LIMIT 0 1000000 TIMEOUT 10000  |  SELECT   file\_id,   file\_name,   doc\_base\_id,   created\_by,   created\_time,   last\_updated\_by,   last\_updated\_time,   COUNT(\*) AS count FROM KM\_2345 WHERE created\_time \>= 0 GROUP BY   file\_id,   file\_name,   doc\_base\_id,   created\_by,   created\_time,   last\_updated\_by,   last\_updated\_time LIMIT 1000000;  |
| Get list of events based on filter |  FT.AGGREGATE KM\_1234 (@EventDate:\[1755144000 1768971599\]) @created\_time:\[0 \+inf\] GROUPBY 1 @doc\_base\_id REDUCE COUNT\_DISTINCT 1 @file\_id AS count\_distinct\_file\_id REDUCE TOLIST 1 @file\_name AS tolist\_file\_name LIMIT 0 1000000 TIMEOUT 10000  |  SELECT   doc\_base\_id,   COUNT(DISTINCT file\_id) AS count\_distinct\_file\_id,   ARRAY\_AGG(DISTINCT file\_name) AS tolist\_file\_name FROM KM\_1234 WHERE EventDate BETWEEN 1755144000 AND 1768971599   AND created\_time \>= 0 GROUP BY doc\_base\_id LIMIT 1000000;  |
| Filter and group count based query |  FT.AGGREGATE KM\_53c4bf8a-8435-4e99-9ec2-e800faf677f3 (@page\_id:{517805590}) @created\_time:\[0 \+inf\] GROUPBY 12 @doc\_base\_id @created\_time @last\_updated\_time @file\_name @file\_id @created\_by @last\_updated\_by @space\_key @title @link @attachment\_file\_name @is\_attachment REDUCE COUNT 0 AS count SORTBY 2 @created\_time DESC LIMIT 0 1000000 TIMEOUT 10000  |  SELECT   doc\_base\_id,   created\_time,   last\_updated\_time,   file\_name,   file\_id,   created\_by,   last\_updated\_by,   space\_key,   title,   link,   attachment\_file\_name,   is\_attachment,   COUNT(\*) AS count FROM KM\_53c4bf8a\_8435\_4e99\_9ec2\_e800faf677f3 WHERE page\_id \= '517805590'   AND created\_time \>= 0 GROUP BY   doc\_base\_id,   created\_time,   last\_updated\_time,   file\_name,   file\_id,   created\_by,   last\_updated\_by,   space\_key,   title,   link,   attachment\_file\_name,   is\_attachment ORDER BY created\_time DESC LIMIT 1000000;  |
| additional examples |  |  |
| ft.search with filters and sorting |  `FT.SEARCH books "((@stock:[(50 +inf] @price:[-inf (20]) @description:(classic))" RETURN 1 title DIALECT 2 LIMIT 0 10`  |  `SELECT title FROM books WHERE stock > 50   AND price < 20   AND description_tsv @@ plainto_tsquery('english', 'classic') LIMIT 10;`  |
| ft.aggregate with filters, reducers, and sorting |  FT.AGGREGATE books "@stock:\[70 \+inf\]" SCORER TFIDF DIALECT 2 GROUPBY 1 @genre REDUCE AVG 1 price AS avg\_price  |  SELECT genre, AVG(price) AS avg\_price FROM books WHERE stock \> 70 GROUP BY genre;  |
| Pure BM25 based test search |  FT.SEARCH books "@description:(thrilling | book | get | lost | beach)" SCORER BM25STD WITHSCORES RETURN 2 title description DIALECT 2 LIMIT 0 20  |  `SELECT   title,   description,   ts_rank(     description_tsv,     plainto_tsquery('english', 'thrilling book get lost beach')   ) AS score FROM books WHERE   description_tsv @@ plainto_tsquery('english', 'thrilling book get lost beach') ORDER BY score DESC LIMIT 20;`   |
| \<more examples\> |  |  |

### **Break down by clause, operator, and datatype:**

* Supported clauses:  
  * SELECT (explicit column list only)  
  * FROM (single index)  
  * WHERE (boolean logic, operators)  
  * ORDER BY  
  * LIMIT / OFFSET  
  * ISMISSING / EXISTS  
  * GROUP BY  
    * With supported [reducers](https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/aggregations/)  
      * COUNT  
      * COUNT\_DISTINCT  
      * SUM  
      * MIN  
      * MAX  
      * AVG  
      * STDDEV  
      * QUANTILE  
      * TOLIST  
      * FIRSTVALUE  
* Supported operators:  
  * \=, \!=  
  * \<, \<=, \>, \>=  
  * IN  
  * AND, OR, NOT  
* Supported data types:  
  * TAG  
  * NUMERIC  
  * TEXT  
  * VECTOR  
  * DATE  
  * GEO

# **Deliverables**

Per the objective, the main deliverable of this work will be a redisvl class allowing for the easy translation between in scope SQL queries and Redis search equivalents. It will be similar if not directly extended from redisvl.query.BaseQuery source code available [here](https://github.com/redis/redis-vl-python/blob/82776afc450818d4358cee7e6071eb5c0eacc2d9/redisvl/query/query.py#L25-L26).

# **Advanced queries (i.e no standard SQL equivalent)**

For vector and other types of queries there may not be direct SQL equivalent statements. For these cases there needs to be agreed upon convention or agreement that the team adopt the client pattern.

### RedisVL client-based example:

```py
from redisvl.query import HybridQuery

user_query = "Thrilling book that I can get lost in at the beach"
vector = hf.embed(user_query, as_buffer=True)

query = HybridQuery(
	text=user_query,
	text_field_name="description",
	vector=vector,
	vector_field_name="vector",
	combination_method="LINEAR",
	yield_text_score_as="text_score",
	yield_vsim_score_as="vector_similarity",
	yield_combined_score_as="hybrid_score",
	return_fields=["title"],
)

results = index.query(query)
```

### Illustrative SQL-translation examples:

| Redis functionality | Redis example | SQL equivalent (Illustrative) |
| ----- | ----- | ----- |
| Vector search with filters and sorting |  FT.SEARCH books "(@genre:{Science\\\\ Fiction} @price:\[-inf 20\])=\>\[KNN 3 @vector $vector AS vector\_distance\]" RETURN 3 title genre vector\_distance SORTBY vector\_distance ASC DIALECT 2 LIMIT 0 3 PARAMS 2 vector \<384-dimension embedding binary data\>  |  `SELECT   title,   genre,   embedding <=> :query_vector AS vector_distance FROM books WHERE genre = 'Science Fiction'   AND price <= 20 ORDER BY embedding <=> :query_vector LIMIT 3;`  |
| Hybrid query BM25 \+ vector |  FT.HYBRID books \# Text Search Component SEARCH "(\~@description:(thrilling | book | get | lost | beach))" SCORER BM25STD YIELD\_SCORE\_AS text\_score \# Vector Search Component VSIM @vector $vector YIELD\_SCORE\_AS vector\_similarity \# Score Combination COMBINE LINEAR 6 ALPHA 0.3 \# text weight BETA 0.7 \# vector weight YIELD\_SCORE\_AS hybrid\_score \# Output LOAD 1 @title LIMIT 0 10 PARAMS 2 vector \<384-dimension embedding binary\>  |  `SELECT   title,   HYBRID_SCORE(     ts_rank(description, plainto_tsquery(:q)),     vector <=> :query_vector,     TEXT_WEIGHT 0.3,     VECTOR_WEIGHT 0.7   ) AS hybrid_score FROM books WHERE description @@ plainto_tsquery(:q) ORDER BY hybrid_score DESC LIMIT 10;`  |
| Aggregate Hybrid search with filters and sorting (pre 8.4) |  FT.AGGREGATE books "(\~@description:(thrilling | book | get | lost | beach))=\>\[KNN 20 @vector $vector AS vector\_distance\]" SCORER BM25 ADDSCORES LOAD 2 title description DIALECT 2 APPLY "(2 \- @vector\_distance) / 2" AS vector\_similarity APPLY "@\_\_score" AS text\_score APPLY "0.3 \* @text\_score \+ 0.7 \* @vector\_similarity" AS hybrid\_score SORTBY 2 @hybrid\_score DESC MAX 20 PARAMS 2 vector \<384-dimension embedding binary\>  |   `SELECT   title,   HYBRID_SCORE(     ts_rank(description, plainto_tsquery(:q)),     vector <=> :query_vector,     TEXT_WEIGHT 0.3,     VECTOR_WEIGHT 0.7   ) AS hybrid_score FROM books WHERE description @@ plainto_tsquery(:q) ORDER BY hybrid_score DESC LIMIT 10;`   |


