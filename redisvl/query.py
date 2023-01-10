import typing as t

from redis.commands.search.query import Query


def create_vector_query(
    return_fields: t.List[str],
    search_type: str = "KNN",
    number_of_results: int = 20,
    vector_field_name: str = "vector",
    vector_param_name: str = "vector",
    return_score: bool = True,
    sort: bool = True,
    tags: str = "*",
    ) -> Query:
    """Create a vector query for use with a SearchIndex

    Args:
        return_fields (t.List[str]): A list of fields to return in the query results
        search_type (str, optional): The type of search to perform. Defaults to "KNN".
        number_of_results (int, optional): The number of results to return. Defaults to 20.
        vector_field_name (str, optional): The name of the vector field in the index. Defaults to "vector".
        vector_param_name (str, optional): The name of the query param for searches. Defaults to "vector".
        return_score (bool, optional): Whether to return the score in the query results. Defaults to True.
        sort (bool, optional): Whether to sort the results by score. Defaults to True.
        tags (str, optional): tag string to filter the results by. Defaults to "*".

    example usage:
        vector_param = "user_vector"
        query = create_vector_query(
            return_fields=["users", "age", "job", "credit_score"],
            search_type="KNN",
            number_of_results=3,
            vector_field_name="user_embedding",
            vector_param_name="user_vector",
            tags="*")
        index.search(query, query_params={"user_vector": query_vector})

    Returns:
        Query: A Query object that can be used with SearchIndex.search
    """
    base_query = f"{tags}=>[{search_type} {number_of_results} @{vector_field_name} ${vector_param_name} AS vector_score]"
    if return_score:
        return_fields.append("vector_score")
    query = Query(base_query).return_fields(*return_fields).dialect(2)
    if sort:
        query.sort_by("vector_score")
    return query