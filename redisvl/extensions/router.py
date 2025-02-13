from pydantic import root_validator


class SemanticRouter(BaseModel):
    # existing fields...
    vectorizer: Optional[HFTextVectorizer] = None
    dtype: str = "float32"

    @root_validator
    def check_vectorizer_dtype(cls, values):
        router_dtype = values.get("dtype")
        vectorizer = values.get("vectorizer")
        if vectorizer is not None and vectorizer.dtype != router_dtype:
            raise ValueError(
                f"Mismatched vectorizer dtype: {vectorizer.dtype} does not match router dtype: {router_dtype}"
            )
        return values
