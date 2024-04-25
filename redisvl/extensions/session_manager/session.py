from typing import Any, Dict, List, Optional, Tuple, Union
from redis import Redis

class SessionManager():
    def __init__(self,
                 from_file: Optional[str] = None,
                 vectorizer: Vectorizer = None
                 redis_client: Optional[Redis] = None,
                 redis_url: str = "redis://localhost:6379"
                 ):
        """ Initialize session memory with index

        Session Manager stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Session history is stored in prompt:response pairs referred to
        as exchanges.

        Args:
            from_file Optional[str]: File to intiaialize the session index with.
            vectorizer Vectorizer: The vectorizer to create embeddings with.
            redis_client Optional[Redis]: A Redis client instance. Defaults to
                None.
            redis_url Optional[str]: The redis url if no Redis client instance
                is provided.  Defaults to "redis://localhost:6379".


        The proposed schema will support a single combined vector embedding
        constructed from the prompt & response in a single string.

        """

        schema = IndexSchema.from_dict({"index": {"name": name, "prefix": prefix}})
        schema.add_fields(
            [
                {"name": prompt_field, "type": "text"},
                {"name": response_field, "type": "text"},
                {"name": timestamp, "type": "numeric"},
                {"name": user_id, "type": "tag"},
                {
                    "name": combined_vector_field,
                    "type": "vector",
                    "attrs": {
                        "dims": vectorizer_dims,
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                 },
            ]
        )

        self._index = SearchIndex(schema=schema)

        if redis_client:
            self._index.set_client(redis_client)
        else:
            self._index.connect(redis_url=redis_url)

        self._index.create()


    def clear(self):
        """ Clears the chat session history. """
        pass


    def fetch_context( self, prompt: str, as_text: bool = False, top_k: int = 5) -> Union[str, List[str]]:
        """ Searches the chat history for information semantically related to
        the specified prompt.

        This method uses vector similarity search with a text prompt as input.
        It checks for semantically similar prompt:response pairs and fetches
        the top k most relevant previous prompt:response pairs to include as
        context to the next LLM call.

        Args:
            prompt str: The text prompt to search for in session memory
            as_text bool: Whether to return the prompt:response pairs as text
            or as a list.
            top_k int: The number of previous exchanges to return. Default is 5.

        Returns:
            Union[str, List[str]: Either a single string, or a list of strings
            containing the most relevant
                return fields for each similar cached response.

        Raises:
            ValueError: If top_k is an invalid integer.
        """
        pass


    def conversation_history(self, as_text: bool = False) -> Union[str, List[str]]:
        """ Retreive the full conversation history in sequential order.

        Args:
            as_text bool: Whether to return the conversation as a single string,
                          or list of alternating prompts and responses.

        Returns:
            Union[str, List[str]]: A single string transcription of the session
                                   or list of strings if as_text is false.
        """
        pass


    def _order_by(self, exchanges: List[str], recency: bool = True) ->  List[str]:
        """ Orders the fetched conversational context by either recency or relevance.

        Args:
            exchanges List[str]: The list of fetched conversation subsections
            recency bool: Whether to sort exchanges by recency or relevance

        Returns:
            List[str]: The re-ordered conversation subsections.
        """
        # need to do this ordering in the query with a timestamp attr
        # because once we get prompt:response strings back the timestamp is stripped
        pass


    @property
    def distance_threshold(self):
        return self.distance_threshold


    def set_distance_threshold(self, threshold):
        self.set_distance_threshold = threshold


    def summarize(self) -> str:
        """ Summarizes the current session into a single string."""
        pass


    def load_previous_sessions(self):
        """ Checks the Redis instance for previous sessions from this user and
            loads them into session memory if they exist.
        """
        pass


    def store(self, exchange: Tuple[str, str]):
        """ Insert a prompt:response pair into the session memory. A timestamp
            is associated with each exchange so that they can be later sorted
            in sequential ordering after retrieval.

            Args:
                exchange Tuple[str, str]: The user prompt and corresponding LLM response.
        """
        pass
