from typing import Any

from redis import Redis

from redisvl.extensions.constants import (
    CONTENT_FIELD_NAME,
    EXTERNAL_INDEX_DROP_CONFLICT,
    ID_FIELD_NAME,
    METADATA_FIELD_NAME,
    ROLE_FIELD_NAME,
    SESSION_FIELD_NAME,
    TIMESTAMP_FIELD_NAME,
    TOOL_FIELD_NAME,
)
from redisvl.extensions.message_history import BaseMessageHistory
from redisvl.extensions.message_history.schema import ChatMessage, MessageHistorySchema
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import CountQuery, FilterQuery
from redisvl.query.filter import Tag
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.utils.log import get_logger
from redisvl.utils.utils import serialize

logger = get_logger(__name__)


class MessageHistory(BaseMessageHistory):

    def __init__(
        self,
        name: str,
        session_tag: str | None = None,
        prefix: str | None = None,
        redis_client: Redis | None = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: dict[str, Any] | None = None,
        create_index: bool = True,
        **kwargs,
    ):
        """Initialize message history

        Message History stores the current and previous user text prompts and
        LLM responses to allow for enriching future prompts with session
        context. Message history is stored in individual user or LLM prompts and
        responses.

        Args:
            name (str): The name of the message history index.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
            prefix (Optional[str]): Prefix for the keys for this conversation data.
                Defaults to None and will be replaced with the index name.
            redis_client (Optional[Redis]): A Redis client instance. Defaults to
                None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.
            create_index (bool): Whether RedisVL creates the index. When False
                the constructor issues no index command at all: the index must
                already exist over this name and prefix. This class never
                validates an existing index's schema, so nothing further is
                verified either way -- and as elsewhere, a live index whose
                prefix or storage type differs from this one is not detected and
                produces empty results rather than an error. See
                :class:`~redisvl.extensions.cache.llm.SemanticCache` for a worked
                example, and :doc:`/user_guide/installation` for the ACL details.
                Defaults to True.

        """
        connection_kwargs = connection_kwargs or {}
        super().__init__(name, session_tag)

        prefix = prefix or name

        schema = MessageHistorySchema.from_params(name, prefix)

        self._index = SearchIndex(
            schema=schema,
            redis_client=redis_client,
            redis_url=redis_url,
            connection_kwargs=connection_kwargs or None,
        )

        self._create_index = create_index
        self._aindex: AsyncSearchIndex | None = None
        if create_index:
            self._index.create(overwrite=False)
        else:
            logger.debug(
                f"create_index=False: assuming index {name!r} exists over prefix "
                f"{prefix!r}."
            )

        self._default_session_filter = Tag(SESSION_FIELD_NAME) == self._session_tag

    def __repr__(self) -> str:
        return f"MessageHistory(name={self._name!r}, session_tag={self._session_tag!r})"

    def clear(self) -> None:
        """Delete every message, leaving the index in place.

        Clears by index membership, so it removes the documents the live index
        covers. Available under ``create_index=False``; dropping the index is
        :meth:`delete`.

        Warning:
            Under ``create_index=False`` the live index is unverified, so if its
            prefix differs from this instance's it removes documents this
            instance never wrote. See :doc:`/user_guide/installation`.
        """
        self._index.clear()

    async def aclear(self) -> None:
        """Async version of :meth:`clear`.

        See :meth:`aget_recent` for the connection caveats.
        """
        aindex = self._get_async_index()
        await aindex.clear()

    def delete(self) -> None:
        """Remove every message and drop the search index.

        Raises:
            ValueError: If ``create_index=False``. Use :meth:`clear` to
                remove the messages and leave the index standing.
        """
        if not self._create_index:
            raise ValueError(EXTERNAL_INDEX_DROP_CONFLICT)
        self._index.delete(drop=True)

    def drop(self, id: str | None = None) -> None:
        """Remove a specific exchange from the conversation history.

        Args:
            id (Optional[str]): The id of the message entry to delete.
                If None then the last entry is deleted.
        """
        if id is None:
            id = self.get_recent(top_k=1, raw=True)[0][ID_FIELD_NAME]  # type: ignore

        self._index._redis_client.delete(self._index.key(id))

    async def adrop(self, id: str | None = None) -> None:
        """Async version of :meth:`drop`.

        See :meth:`aget_recent` for the connection caveats.
        """
        aindex = self._get_async_index()
        if id is None:
            recent = await self.aget_recent(top_k=1, raw=True)
            id = recent[0][ID_FIELD_NAME]  # type: ignore

        await aindex.drop_keys(aindex.key(id))

    def _count_query(self, session_tag: str | None) -> CountQuery:
        return CountQuery(
            filter_expression=(
                Tag(SESSION_FIELD_NAME) == session_tag
                if session_tag
                else self._default_session_filter
            )
        )

    def count(self, session_tag=None):
        return self._index.query(self._count_query(session_tag))

    async def acount(self, session_tag=None):
        """Async version of :meth:`count`.

        See :meth:`aget_recent` for the connection caveats.
        """
        aindex = self._get_async_index()
        return await aindex.query(self._count_query(session_tag))

    @property
    def messages(self) -> list[str] | list[dict[str, str]]:
        """Returns every message in the current session, oldest first.

        The result is not bounded. For a bounded read of the latest messages,
        use ``get_recent(top_k=...)``.
        """
        # TODO raw or as_text?
        # TODO refactor this method to use get_recent and support other session tags?
        return_fields = [
            ID_FIELD_NAME,
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TOOL_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        query = FilterQuery(
            filter_expression=self._default_session_filter,
            return_fields=return_fields,
            num_results=self.count(),
        )
        query.sort_by(TIMESTAMP_FIELD_NAME, asc=True)
        messages = self._index.query(query)

        return self._format_context(messages, as_text=False)

    def get_recent(
        self,
        top_k: int = 5,
        as_text: bool = False,
        raw: bool = False,
        session_tag: str | None = None,
        role: str | list[str] | None = None,
    ) -> list[str] | list[dict[str, str]]:
        """Retrieve the recent message history in sequential order.

        Args:
            top_k (int): The number of previous messages to return. Default is 5.
            as_text (bool): Whether to return the conversation as a single string,
                or list of alternating prompts and responses.
            raw (bool): Whether to return the full Redis hash entry or just the
                prompt and response.
            session_tag (Optional[str]): Tag of the entries linked to a specific
                conversation session. Defaults to instance ULID.
            role (Optional[Union[str, List[str]]]): Filter messages by role(s).
                Can be a single role string ("system", "user", "llm", "tool") or
                a list of roles. If None, all roles are returned.

        Returns:
            Union[str, List[str]]: A single string transcription of the messages
                or list of strings if as_text is false.

        Raises:
            ValueError: if top_k is not an integer greater than or equal to 0,
                or if role contains invalid values.
        """
        query = self._recent_query(top_k, session_tag, role)
        messages = self._index.query(query)
        return self._recent_result(messages, as_text, raw)

    async def aget_recent(
        self,
        top_k: int = 5,
        as_text: bool = False,
        raw: bool = False,
        session_tag: str | None = None,
        role: str | list[str] | None = None,
    ) -> list[str] | list[dict[str, str]]:
        """Async version of :meth:`get_recent`: the same query on an
        :class:`~redisvl.index.AsyncSearchIndex`, without blocking the loop.

        Pass ``redis_url`` rather than a client to the history if you need the
        async methods over Sentinel or Redis Cluster: those clients cannot be
        converted to their async form. The async client is
        bound to the event loop that made the first call, so call
        :meth:`adisconnect` before switching loops.
        """
        query = self._recent_query(top_k, session_tag, role)
        aindex = self._get_async_index()
        messages = await aindex.query(query)
        return self._recent_result(messages, as_text, raw)

    def _recent_query(
        self,
        top_k: int,
        session_tag: str | None,
        role: str | list[str] | None,
    ) -> FilterQuery:
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be an integer greater than or equal to 0")

        # Validate and normalize role parameter
        roles_to_filter = self._validate_roles(role)

        return_fields = [
            ID_FIELD_NAME,
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TOOL_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        session_filter = (
            Tag(SESSION_FIELD_NAME) == session_tag
            if session_tag
            else self._default_session_filter
        )

        # Combine session filter with role filter if provided
        filter_expression = session_filter
        if roles_to_filter is not None:
            if len(roles_to_filter) == 1:
                role_filter = Tag(ROLE_FIELD_NAME) == roles_to_filter[0]
            else:
                # Multiple roles - use OR logic
                role_filters = [Tag(ROLE_FIELD_NAME) == r for r in roles_to_filter]
                role_filter = role_filters[0]
                for rf in role_filters[1:]:
                    role_filter = role_filter | rf

            filter_expression = session_filter & role_filter

        query = FilterQuery(
            filter_expression=filter_expression,
            return_fields=return_fields,
            num_results=top_k,
        )
        query.sort_by(TIMESTAMP_FIELD_NAME, asc=False)
        return query

    def _recent_result(
        self, messages: list[dict[str, Any]], as_text: bool, raw: bool
    ) -> list[str] | list[dict[str, str]]:
        if raw:
            return messages[::-1]
        return self._format_context(messages[::-1], as_text)

    def store(self, prompt: str, response: str, session_tag: str | None = None) -> None:
        """Insert a prompt:response pair into the message history. A timestamp
        is associated with each exchange so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            prompt (str): The user prompt to the LLM.
            response (str): The corresponding LLM response.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
        """
        self.add_messages(self._exchange(prompt, response), session_tag)

    async def astore(
        self, prompt: str, response: str, session_tag: str | None = None
    ) -> None:
        """Async version of :meth:`store`.

        See :meth:`aget_recent` for the connection caveats.
        """
        await self.aadd_messages(self._exchange(prompt, response), session_tag)

    @staticmethod
    def _exchange(prompt: str, response: str) -> list[dict[str, str]]:
        return [
            {ROLE_FIELD_NAME: "user", CONTENT_FIELD_NAME: prompt},
            {ROLE_FIELD_NAME: "llm", CONTENT_FIELD_NAME: response},
        ]

    def add_messages(
        self, messages: list[dict[str, str]], session_tag: str | None = None
    ) -> None:
        """Insert a list of prompts and responses into the message history.
        A timestamp is associated with each so that they can be later sorted
        in sequential ordering after retrieval.

        Args:
            messages (List[Dict[str, str]]): The list of user prompts and LLM responses.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
        """
        chat_messages = self._chat_messages(messages, session_tag)
        self._index.load(data=chat_messages, id_field=ID_FIELD_NAME)

    async def aadd_messages(
        self, messages: list[dict[str, str]], session_tag: str | None = None
    ) -> None:
        """Async version of :meth:`add_messages`.

        See :meth:`aget_recent` for the connection caveats.
        """
        chat_messages = self._chat_messages(messages, session_tag)
        aindex = self._get_async_index()
        await aindex.load(data=chat_messages, id_field=ID_FIELD_NAME)

    def _chat_messages(
        self, messages: list[dict[str, str]], session_tag: str | None
    ) -> list[dict[str, Any]]:
        session_tag = session_tag or self._session_tag
        chat_messages: list[dict[str, Any]] = []

        for message in messages:

            chat_message = ChatMessage(
                role=message[ROLE_FIELD_NAME],
                content=message[CONTENT_FIELD_NAME],
                session_tag=session_tag,
            )

            if TOOL_FIELD_NAME in message:
                chat_message.tool_call_id = message[TOOL_FIELD_NAME]
            if METADATA_FIELD_NAME in message:
                chat_message.metadata = serialize(message[METADATA_FIELD_NAME])
            chat_messages.append(chat_message.to_dict())

        return chat_messages

    def add_message(
        self, message: dict[str, str], session_tag: str | None = None
    ) -> None:
        """Insert a single prompt or response into the message history.
        A timestamp is associated with it so that it can be later sorted
        in sequential ordering after retrieval.

        Args:
            message (Dict[str,str]): The user prompt or LLM response.
            session_tag (Optional[str]): Tag to be added to entries to link to a specific
                conversation session. Defaults to instance ULID.
        """
        self.add_messages([message], session_tag)

    async def aadd_message(
        self, message: dict[str, str], session_tag: str | None = None
    ) -> None:
        """Async version of :meth:`add_message`.

        See :meth:`aget_recent` for the connection caveats.
        """
        await self.aadd_messages([message], session_tag)

    def _get_async_index(self) -> AsyncSearchIndex:
        """Lazily construct the async search index over the same Redis index."""
        if self._aindex is None:
            index = self._index
            if index._owns_redis_client and index._redis_url:
                self._aindex = AsyncSearchIndex(
                    schema=index.schema,
                    redis_url=index._redis_url,
                    connection_kwargs=index._connection_kwargs or None,
                    lib_name=index._lib_name,
                )
            else:
                self._aindex = AsyncSearchIndex(
                    schema=index.schema,
                    redis_client=RedisConnectionFactory.sync_to_async_redis(
                        index._redis_client  # type: ignore[arg-type]
                    ),
                    owns_client=True,
                )
        return self._aindex

    @property
    def aindex(self) -> AsyncSearchIndex | None:
        """The AsyncSearchIndex used by the async methods, once created.

        Returns:
            AsyncSearchIndex: The async search index, or None before the first
            async call.
        """
        return self._aindex

    async def adisconnect(self) -> None:
        """Close the async Redis connection opened by the async methods."""
        if self._aindex is not None:
            await self._aindex.disconnect()
            self._aindex = None
