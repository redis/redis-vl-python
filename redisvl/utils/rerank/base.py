from redis.asyncio import Redis
from redisvl.schema import IndexSchema
from redisvl.index import AsyncSearchIndex
from redisvl.utils.vectorize import CohereTextVectorizer
from redisvl.utils.rerank import CohereReranker
from redisvl.query import VectorQuery


vectorizer = CohereTextVectorizer()
reranker = CohereReranker(
    model="rerank-english-v2.0", limit=5, rank_by="", max_chunks_per_doc
)
# when there's an overflow from context length

client = Redis.from_url("redis://localhost:6379")
schema = IndexSchema.from_yaml("schema/schema.yaml")


async def main(data, query: str):
    """To start"""
    index = AsyncSearchIndex(schema, client)

    await index.create(overwrite=True, drop=True)
    await index.load([data])

    vector_query = VectorQuery(
        vector=vectorizer.embed(query),
        vector_field_name="",
        return_fields=[],
        num_results=20
    )

    # TODO think about the scoring implementation
    # add score to the dict
    results = await index.query(vector_query)
    ranked_results = await reranker.rank(
        query, results, limit=4, rank_by="", return_score=True
    )
    # How do we handle multiple fields for overflow?
    # If you do provide multiple fields, truncate in order?
    # Support single field to start
    return ranked_results




async def main(data, query: str):
    """Maybe in the future???"""
    index = AsyncSearchIndex(
        schema,
        client,
        vectorizer=vectorizer,
        reranker=reranker
    )

    await index.create(overwrite=True, drop=True)
    await index.load([data])

    vector_query = VectorQuery(
        vector_field_name="",
        return_fields=[],
        num_results=20
    )

    results = await index.pipeline_run(
        vectorizer,
        vector_query,
        reranker
    )

