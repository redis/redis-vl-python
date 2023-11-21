# Vector Databases

Vector databases are specifically designed to store, retrieve and search over dense numerical vectors in an efficient manner. While traditional databases typically organize data in rows and columns, Vector databases cater to applications leveraging image recognition, natural language processing, and recommendation systems, where data is represented as [Vector Embeddings](VectorEmbeddings.md) in a multi-dimensional space.

They are designed to efficiently handle the storage and retrieval of these dense numerical Vectors. They leverage specialized data structures and indexing techniques, such as [hierarchical navigable small world (HNSW)](VectorIndex.md) and product quantization, to enable swift similarity and semantic searches. These databases enable users to find Vectors that are most similar to a given query vector based on a chosen [Distance Metric](DistanceMetrics.md), such as euclidean distance, cosine similarity, or dot product.

In summary, vector databases are purpose-built databases that excel at storing and querying over dense numerical vectors efficiently. They enable fast similarity searches, accommodate complex data types, and offer advantages in various domains where similarity search and real-time analytics on unstructured are critical.

## How does a vector database function?

The operation of a vector database differs significantly from that of traditional databases, which primarily store scalar data in rows and columns. In contrast, vector databases operate on vectors and are optimized and queried in a distinct manner.

A vector database utilizes various algorithms to enable Approximate Nearest Neighbor (ANN) search. These algorithms are carefully selected and combined to enhance the efficiency of the search process. Techniques such as hashing, quantization, and graph-based search are employed to optimize the retrieval of neighboring vectors.

They employ various algorithms organized into a pipeline to efficiently locate the nearest neighbors to a given query vector. While it is true that ANN approaches within vector databases prioritize speed over exactness, it is important to note that this does not apply to all vector databases. In the case of a K-nearest neighbor (KNN) approach like FLAT, exactness is guaranteed but at the cost of additional latency. However, through thoughtful system design, it is possible to strike a balance and achieve near-perfect accuracy while still delivering ultra-fast search capabilities in vector databases. By optimizing the pipeline and considering factors such as indexing techniques, distance metrics, and query processing methods, vector databases can provide both speed and accuracy tailored to specific use cases.

Let's delve into a typical architecture for a vector database:

![VDB Architecture](../../assets/vdb_architect.png)

Indexing: The process of [indexing vectors](VectorIndex.md) in a vector database involves utilizing algorithms such as PQ (Product Quantization), LSH (Locality-Sensitive Hashing), HNSW or FLAT. This step essentially maps the vectors to a data structure that facilitates faster searching.

Pre-filtering: Prior to vector search, pre-filtering techniques can be applied to narrow down the dataset. This may involve performing filtering operations, range queries, or set intersections to refine the dataset and focus on relevant vectors.

Querying: When it comes to querying, the vector database compares the query vector, which has been indexed, with the vectors stored in the dataset to determine the nearest neighbors. This comparison involves applying the specific similarity metric used by the index.

Post Processing: In certain cases, after identifying the nearest neighbors, the vector database retrieves them from the dataset and carries out additional processing steps like post-filtering, aggregations, or re-ranking using a Vector Database
