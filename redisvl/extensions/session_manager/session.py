import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from redis import Redis

from redisvl.index import SearchIndex
from redisvl.query import RangeQuery
from redisvl.redis.utils import array_to_buffer
from redisvl.schema.schema import IndexSchema
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer

class SessionManager:
    _session_id: str = "any_session"
    _user_id: str = "any_user"
    _application_id: str = "any_app"

    def __init__(self,
                 name: str = "session", 
                 prefix: Optional[str] = None,
                 session_id: str = None,
                 user_id: str = None,
                 application_id: str = None,
                 vectorizer: Optional[BaseVectorizer] = None,
                 redis_client: Optional[Redis] = None,
                 redis_url: str = "redis://localhost:6379"
                 ):
        """ Initialize session memory with index

        Session Manager stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Session history is stored in prompt:response pairs referred to
        as exchanges.

        Args:
            name str:
            prefix Optional[str]: 
            vectorizer Vectorizer: The vectorizer to create embeddings with.
            redis_client Optional[Redis]: A Redis client instance. Defaults to
                None.
            redis_url Optional[str]: The redis url if no Redis client instance
                is provided.  Defaults to "redis://localhost:6379".


        The proposed schema will support a single combined vector embedding
        constructed from the prompt & response in a single string.

        """
        if not prefix:
            prefix = name
        if not session_id:
            self._session_id = "any_session"
        if not user_id:
            self._user_id = "any_id"
        if not application_id:
            self._application_id = "any_app"

        if vectorizer is None:
            self._vectorizer = HFTextVectorizer(
                model="sentence-transformers/all-mpnet-base-v2"
            )

        schema = IndexSchema.from_dict({"index": {"name": name, "prefix": prefix}})

        schema.add_fields(
            [
                {"name": "prompt", "type": "text"},
                {"name": "response", "type": "text"},
                {"name": "timestamp", "type": "numeric"},
                {"name": self._session_id, "type": "tag"},
                {"name": self._user_id, "type": "tag"},
                {"name": self._application_id, "type": "tag"},
                {"name": "token_count", "type": "numeric"},
                {
                    "name": "combined_vector_field",
                    "type": "vector",
                    "attrs": {
                        "dims": self._vectorizer.dims,
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
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


    def fetch_context(self, prompt: str, as_text: bool = False, top_k: int = 5) -> Union[str, List[str]]:
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
       
        return_fields = [
            self._session_id,
            self._user_id,
            self._application_id,
            "prompt",
            "response",
            "combined_vector_field",
        ]


        query = RangeQuery(
            vector=self._vectorizer.embed(prompt),
            vector_field_name="combined_vector_field",
            return_fields=return_fields,
            distance_threshold=0.2, #self._distance_threshold
            num_results=top_k,
            return_score=True
        )
        
        hits = self._index.query(query)
        return hits


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
        return self._distance_threshold


    def set_distance_threshold(self, threshold):
        self._distance_threshold = threshold


    def summarize(self) -> str:
        """ Summarizes the current session into a single string."""
        pass


    def load_previous_sessions(self):
        """ Checks the Redis instance for previous sessions from this user and
            loads them into session memory if they exist.
        """
        pass


    def store(self,
              exchange: Tuple[str, str],
              scope: str,
              session_id: Union[str, int] = None,
              user_id: Union[str, int] = None,
              application_id: Union[str, int] = None,
              ):
        """ Insert a prompt:response pair into the session memory. A timestamp
            is associated with each exchange so that they can be later sorted
            in sequential ordering after retrieval.

            Args:
                exchange Tuple[str, str]: The user prompt and corresponding LLM
                    response.
                scope str: the scope of access this exchange can be retrieved by.
                    must be one of {Session, User, Application}.
                session_id Union[str, int]: = the session id tag to index this
                    exchange with. Must be provided if scope==Session.
                user_id Union[str, int]: the user id tag to index this exchange
                    with. Must be provided if scope==User
                application_id Union[str, int]: = the application id tag to
                    index this exchange with. Must be provided if scope==Application
        """
        
        vector = self._vectorizer.embed(exchange[0] + exchange[1])
        timestamp = int(datetime.now().timestamp())
        payload = {
                #TODO decide if hash id should be based on prompt, response,
                # user, session, app, or some combination 
                "id": self.hash_input(exchange[0]+str(timestamp)),
                "prompt": exchange[0],
                "response": exchange[1],
                "timestamp": timestamp,
                "session_id": session_id or self._session_id,
                "user_id": user_id or self._user_id,
                "application_id": application_id or self._application_id,
                "token_count": 1, #TODO get actual token count
                "combined_vector_field": array_to_buffer(vector)
        }
        '''
                {"name": "prompt", "type": "text"},
                {"name": "response", "type": "text"},
                {"name": "timestamp", "type": "numeric"},
                {"name": session_id, "type": "tag"},
                {"name": user_id, "type": "tag"},
                {"name": application_id, "type": "tag"},
                {"name": "token_count", "type": "numeric"},
                {
                    "name": "combined_vector_field",
                    "type": "vector",
                    "attrs": {
                        "dims": self._vectorizer.dims,
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                 }
        '''
        keys = self._index.load(data=[payload])
        return keys


    def set_preamble(self, prompt: str) -> None:
        """ Add a preamble statement to the the begining of each session history
            and will be included in each subsequent LLM call.
        """ 
        self._preamble = prompt # TODO store this in Redis with asigned scope?


    def timstamp_to_int(self, timestamp: datetime.timestamp) -> int: 
        """ Converts a datetime object into integer for storage as numeric field
            in hash.
        """
        pass


    def int_to_timestamp(self, epoch_time: int) -> datetime.timestamp: 
        """ Converts a numeric date expressed in epoch time into datetime 
            object.
        """
        pass


    def hash_input(self, prompt: str):
        """Hashes the input using SHA256."""
        #TODO find out if this is really necessary
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

