import typing as t

from redis.commands.search.query import Query


def create_vector_query(
    return_fields: t.List[str],
    search_type: str = "KNN",
    number_of_results: int = 20,
    vector_field_name: str = "vector",
    tags: str = "*",
):
    base_query = f"{tags}=>[{search_type} {number_of_results} @{vector_field_name} $vector AS vector_score]"
    return (
        Query(base_query)
        .sort_by("vector_score")
        .paging(0, number_of_results)
        .return_fields(*return_fields)
        .dialect(2)
    )
