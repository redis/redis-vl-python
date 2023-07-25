from typing import Any, Dict, List, Optional

import numpy as np
from redis.commands.search.query import Query

from redisvl.utils.utils import TokenEscaper, array_to_buffer


class Filter:
    escaper = TokenEscaper()

    def __init__(self, field):
        self._field = field
        self._filters = []

    def __str__(self):
        base = "(" + self.to_string()
        if self._filters:
            base += " ".join(self._filters)
        return base + ")"

    def __iadd__(self, other):
        "intersection '+='"
        self._filters.append(f" {other.to_string()}")
        return self

    def __iand__(self, other):
        "union '&='"
        self._filters.append(f" |{other.to_string()}")
        return self

    def __isub__(self, other):
        "subtract '-='"
        self._filters.append(f" -{other.to_string()}")
        return self

    def __ixor__(self, other):
        "With optional '^='"
        self._filters.append(f" ~{other.to_string()}")
        return self

    def to_string(self) -> str:
        raise NotImplementedError


class TagFilter(Filter):
    def __init__(self, field, tags: List[str]):
        super().__init__(field)
        self.tags = tags

    def to_string(self) -> str:
        """Converts the tag filter to a string.

        Returns:
            str: The tag filter as a string.
        """
        if not isinstance(self.tags, list):
            self.tags = [self.tags]
        return (
            "@"
            + self._field
            + ":{"
            + " | ".join([self.escaper.escape(tag) for tag in self.tags])
            + "}"
        )


class NumericFilter(Filter):

    def __init__(self, field, minval, maxval, minExclusive=False, maxExclusive=False):
        """Filter for Numeric fields.

        Args:
            field (str): The field to filter on.
            minval (int): The minimum value.
            maxval (int): The maximum value.
            minExclusive (bool, optional): Whether the minimum value is exclusive. Defaults to False.
            maxExclusive (bool, optional): Whether the maximum value is exclusive. Defaults to False.
        """
        self.top = maxval if not maxExclusive else f"({maxval}"
        self.bottom = minval if not minExclusive else f"{minval})"
        super().__init__(field)

    def to_string(self):
        return "@" + self._field + ":[" + str(self.bottom) + " " + str(self.top) + "]"


class TextFilter(Filter):

    def __init__(self, field, text: str):
        """Filter for Text fields.
        Args:
            field (str): The field to filter on.
            text (str): The text to filter on.
        """
        super().__init__(field)
        self.text = text

    def to_string(self) -> str:
        """Converts the filter to a string.

        Returns:
            str: The filter as a string.
        """
        return "@" + self._field + ":" + self.escaper.escape(self.text)


class BaseQuery:
    def __init__(
        self, return_fields: Optional[List[str]] = None, num_results: Optional[int] = 10
    ):
        self._return_fields = return_fields
        self._num_results = num_results

    @property
    def query(self):
        pass

    @property
    def params(self):
        pass


class VectorQuery(BaseQuery):
    dtypes = {
        "float32": np.float32,
        "float64": np.float64,
    }

    def __init__(
        self,
        vector: List[float],
        vector_field_name: str,
        return_fields: List[str],
        hybrid_filter: Filter = None,
        dtype: str = "float32",
        num_results: Optional[int] = 10,
    ):
        """Query for vector fields

        Args:
            vector (List[float]): The vector to query for.
            vector_field_name (str): The name of the vector field
            return_fields (List[str]): The fields to return.
            hybrid_filter (Filter, optional): A filter to apply to the query. Defaults to None.
            dtype (str, optional): The dtype of the vector. Defaults to "float32".
            num_results (Optional[int], optional): The number of results to return. Defaults to 10.
        """
        super().__init__(return_fields, num_results)
        self._vector = vector
        self._field = vector_field_name
        self._dtype = dtype.lower()
        if hybrid_filter:
            self.set_filter(hybrid_filter)
        else:
            self._filter = "*"

    def set_filter(self, hybrid_filter: Filter):
        """Set the filter for the query.

        Args:
            hybrid_filter (Filter): The filter to apply to the query.
        """
        if not isinstance(hybrid_filter, Filter):
            raise TypeError("hybrid_filter must be of type redisvl.query.Filter")
        self._filter = str(hybrid_filter)

    def get_filter(self):
        """Get the filter for the query.

        Returns:
            Filter: The filter for the query.
        """
        return self._filter

    @property
    def query(self):
        """Return a Redis-Py Query object representing the query.

        Returns:
            redis.commands.search.query.Query: The query object.
        """
        base_query = f"{self._filter}=>[KNN {self._num_results} @{self._field} $vector AS vector_distance]"
        query = (
            Query(base_query)
            .return_fields(*self._return_fields)
            .sort_by("vector_distance")
            .paging(0, self._num_results)
            .dialect(2)
        )
        return query

    @property
    def params(self):
        """Return the parameters for the query.

        Returns:
            Dict[str, Any]: The parameters for the query.
        """
        return {"vector": array_to_buffer(self._vector, dtype=self.dtypes[self._dtype])}
