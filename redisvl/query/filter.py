import datetime
import re
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from redisvl.utils.token_escaper import TokenEscaper

# disable mypy error for dunder method overrides
# mypy: disable-error-code="override"


class Inclusive(str, Enum):
    """Enum for valid inclusive options"""

    BOTH = "both"
    """Inclusive of both sides of range (default)"""
    NEITHER = "neither"
    """Inclusive of neither side of range"""
    LEFT = "left"
    """Inclusive of only left"""
    RIGHT = "right"
    """Inclusive of only right"""


class FilterOperator(Enum):
    EQ = 1
    NE = 2
    LT = 3
    GT = 4
    LE = 5
    GE = 6
    OR = 7
    AND = 8
    LIKE = 9
    IN = 10
    BETWEEN = 11


class FilterField:
    escaper: TokenEscaper = TokenEscaper()
    OPERATORS: Dict[FilterOperator, str] = {}

    def __init__(self, field: str):
        self._field = field
        self._value: Any = None
        self._operator: FilterOperator = FilterOperator.EQ

    def equals(self, other: "FilterField") -> bool:
        if not isinstance(other, type(self)):
            return False
        return (self._field == other._field) and (self._value == other._value)

    def _set_value(
        self,
        val: Any,
        val_type: Union[type, Tuple[type, ...]],
        operator: FilterOperator,
    ):
        # check that the operator is supported by this class
        if operator not in self.OPERATORS:
            raise ValueError(
                f"Operator {operator} not supported by {self.__class__.__name__}. "
                + f"Supported operators are {self.OPERATORS.values()}"
            )
        # check that the value is of the proper type
        if not isinstance(val, val_type):
            raise TypeError(
                f"Right side argument passed to operator {self.OPERATORS[operator]} "
                f"with left side "
                f"argument {self.__class__.__name__} must be of type {val_type}"
            )
        self._value = val
        self._operator = operator


def check_operator_misuse(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(instance: Any, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        # Extracting 'other' from positional arguments or keyword arguments
        other = kwargs.get("other") if "other" in kwargs else None
        if not other:
            for arg in args:
                if isinstance(arg, type(instance)):
                    other = arg
                    break

        if isinstance(other, type(instance)):
            raise ValueError(
                "Equality operators are overridden for FilterExpression creation. Use "
                ".equals() for equality checks"
            )
        return func(instance, *args, **kwargs)

    return wrapper


class Tag(FilterField):
    """A Tag filter can be applied to Tag fields"""

    OPERATORS: Dict[FilterOperator, str] = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.IN: "==",
    }
    OPERATOR_MAP: Dict[FilterOperator, str] = {
        FilterOperator.EQ: "@%s:{%s}",
        FilterOperator.NE: "(-@%s:{%s})",
        FilterOperator.IN: "@%s:{%s}",
    }
    SUPPORTED_VAL_TYPES = (list, set, tuple, str, type(None))

    def _set_tag_value(
        self, other: Union[List[str], Set[str], str], operator: FilterOperator
    ):
        if isinstance(other, (list, set, tuple)):
            try:
                # "if val" clause removes non-truthy values from list
                other = [str(val) for val in other if val]
            except ValueError:
                raise ValueError("All tags within collection must be strings")
        # above to catch the "" case
        elif not other:
            other = []
        elif isinstance(other, str):
            other = [other]

        self._set_value(other, self.SUPPORTED_VAL_TYPES, operator)

    @check_operator_misuse
    def __eq__(self, other: Union[List[str], str]) -> "FilterExpression":
        """Create a Tag equality filter expression.

        Args:
            other (Union[List[str], str]): The tag(s) to filter on.

        .. code-block:: python

            from redisvl.query.filter import Tag

            f = Tag("brand") == "nike"
        """
        self._set_tag_value(other, FilterOperator.EQ)
        return FilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other) -> "FilterExpression":
        """Create a Tag inequality filter expression.

        Args:
            other (Union[List[str], str]): The tag(s) to filter on.

        .. code-block:: python

            from redisvl.query.filter import Tag
            f = Tag("brand") != "nike"

        """
        self._set_tag_value(other, FilterOperator.NE)
        return FilterExpression(str(self))

    @property
    def _formatted_tag_value(self) -> str:
        return "|".join([self.escaper.escape(tag) for tag in self._value])

    def __str__(self) -> str:
        """Return the Redis Query string for the Tag filter"""
        if not self._value:
            return "*"

        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            self._formatted_tag_value,
        )


class GeoSpec:
    GEO_UNITS = ["m", "km", "mi", "ft"]

    # class for the operand for FilterExpressions with Geo
    def __init__(self, longitude: float, latitude: float, unit: str = "km"):
        if unit.lower() not in self.GEO_UNITS:
            raise ValueError(f"Unit must be one of {self.GEO_UNITS}")
        self._longitude = longitude
        self._latitude = latitude
        self._unit = unit.lower()


class GeoRadius(GeoSpec):
    """A GeoRadius is a GeoSpec representing a geographic radius."""

    def __init__(
        self,
        longitude: float,
        latitude: float,
        radius: int = 1,
        unit: str = "km",
    ):
        """Create a GeoRadius specification (GeoSpec)

        Args:
            longitude (float): The longitude of the center of the radius.
            latitude (float): The latitude of the center of the radius.
            radius (int, optional): The radius of the circle. Defaults to 1.
            unit (str, optional): The unit of the radius. Defaults to "km".

        Raises:
            ValueError: If the unit is not one of "m", "km", "mi", or "ft".
        """
        super().__init__(longitude, latitude, unit)
        self._radius = radius

    def get_args(self) -> List[Union[float, int, str]]:
        return [self._longitude, self._latitude, self._radius, self._unit]


class Geo(FilterField):
    """A Geo is a FilterField representing a geographic (lat/lon) field in a
    Redis index."""

    OPERATORS: Dict[FilterOperator, str] = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
    }
    OPERATOR_MAP: Dict[FilterOperator, str] = {
        FilterOperator.EQ: "@%s:[%s %s %i %s]",
        FilterOperator.NE: "(-@%s:[%s %s %i %s])",
    }
    SUPPORTED_VAL_TYPES = (GeoSpec, type(None))

    @check_operator_misuse
    def __eq__(self, other) -> "FilterExpression":
        """Create a geographic filter within a specified GeoRadius.

        Args:
            other (GeoRadius): The geographic spec to filter on.

        .. code-block:: python

            from redisvl.query.filter import Geo, GeoRadius

            f = Geo("location") == GeoRadius(-122.4194, 37.7749, 1, unit="m")

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.EQ)  # type: ignore
        return FilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other) -> "FilterExpression":
        """Create a geographic filter outside of a specified GeoRadius.

        Args:
            other (GeoRadius): The geographic spec to filter on.

        .. code-block:: python

            from redisvl.query.filter import Geo, GeoRadius

            f = Geo("location") != GeoRadius(-122.4194, 37.7749, 1, unit="m")

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.NE)  # type: ignore
        return FilterExpression(str(self))

    def __str__(self) -> str:
        """Return the Redis Query string for the Geo filter"""
        if not self._value:
            return "*"

        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            *self._value.get_args(),
        )


class Num(FilterField):
    """A Num is a FilterField representing a numeric field in a Redis index."""

    OPERATORS: Dict[FilterOperator, str] = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.LT: "<",
        FilterOperator.GT: ">",
        FilterOperator.LE: "<=",
        FilterOperator.GE: ">=",
        FilterOperator.BETWEEN: "between",
    }
    OPERATOR_MAP: Dict[FilterOperator, str] = {
        FilterOperator.EQ: "@%s:[%s %s]",
        FilterOperator.NE: "(-@%s:[%s %s])",
        FilterOperator.GT: "@%s:[(%s +inf]",
        FilterOperator.LT: "@%s:[-inf (%s]",
        FilterOperator.GE: "@%s:[%s +inf]",
        FilterOperator.LE: "@%s:[-inf %s]",
        FilterOperator.BETWEEN: "@%s:[%s %s]",
    }

    SUPPORTED_VAL_TYPES = (int, float, tuple, type(None))

    def __eq__(self, other: int) -> "FilterExpression":
        """Create a Numeric equality filter expression.

        Args:
            other (int): The value to filter on.

        .. code-block:: python

            from redisvl.query.filter import Num
            f = Num("zipcode") == 90210

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other: int) -> "FilterExpression":
        """Create a Numeric inequality filter expression.

        Args:
            other (int): The value to filter on.

        .. code-block:: python

            from redisvl.query.filter import Num

            f = Num("zipcode") != 90210

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.NE)
        return FilterExpression(str(self))

    def __gt__(self, other: int) -> "FilterExpression":
        """Create a Numeric greater than filter expression.

        Args:
            other (int): The value to filter on.

        .. code-block:: python

            from redisvl.query.filter import Num

            f = Num("age") > 18

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.GT)
        return FilterExpression(str(self))

    def __lt__(self, other: int) -> "FilterExpression":
        """Create a Numeric less than filter expression.

        Args:
            other (int): The value to filter on.

        .. code-block:: python

            from redisvl.query.filter import Num

            f = Num("age") < 18

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.LT)
        return FilterExpression(str(self))

    def __ge__(self, other: int) -> "FilterExpression":
        """Create a Numeric greater than or equal to filter expression.

        Args:
            other (int): The value to filter on.

        .. code-block:: python

            from redisvl.query.filter import Num

            f = Num("age") >= 18

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.GE)
        return FilterExpression(str(self))

    def __le__(self, other: int) -> "FilterExpression":
        """Create a Numeric less than or equal to filter expression.

        Args:
            other (int): The value to filter on.

        .. code-block:: python

            from redisvl.query.filter import Num

            f = Num("age") <= 18

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.LE)
        return FilterExpression(str(self))

    @staticmethod
    def _validate_inclusive_string(inclusive: str) -> Inclusive:
        try:
            return Inclusive(inclusive)
        except:
            raise ValueError(
                f"Invalid inclusive value must be: {[i.value for i in Inclusive]}"
            )

    def _format_inclusive_between(
        self, inclusive: Inclusive, start: int, end: int
    ) -> str:
        if inclusive.value == Inclusive.BOTH.value:
            return f"@{self._field}:[{start} {end}]"

        if inclusive.value == Inclusive.NEITHER.value:
            return f"@{self._field}:[({start} ({end}]"

        if inclusive.value == Inclusive.LEFT.value:
            return f"@{self._field}:[{start} ({end}]"

        if inclusive.value == Inclusive.RIGHT.value:
            return f"@{self._field}:[({start} {end}]"

        raise ValueError(f"Inclusive value not found")

    def between(
        self, start: int, end: int, inclusive: str = "both"
    ) -> "FilterExpression":
        """Operator for searching values between two numeric values."""
        inclusive = self._validate_inclusive_string(inclusive)
        expression = self._format_inclusive_between(inclusive, start, end)

        return FilterExpression(expression)

    def __str__(self) -> str:
        """Return the Redis Query string for the Numeric filter"""
        if self._value is None:
            return "*"
        if self._operator == FilterOperator.BETWEEN:
            return self.OPERATOR_MAP[self._operator] % (
                self._field,
                self._value[0],
                self._value[1],
            )
        if self._operator == FilterOperator.EQ or self._operator == FilterOperator.NE:
            return self.OPERATOR_MAP[self._operator] % (
                self._field,
                self._value,
                self._value,
            )
        else:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)


class Text(FilterField):
    """A Text is a FilterField representing a text field in a Redis index."""

    OPERATORS: Dict[FilterOperator, str] = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.LIKE: "%",
    }
    OPERATOR_MAP: Dict[FilterOperator, str] = {
        FilterOperator.EQ: '@%s:("%s")',
        FilterOperator.NE: '(-@%s:"%s")',
        FilterOperator.LIKE: "@%s:(%s)",
    }
    SUPPORTED_VAL_TYPES = (str, type(None))

    @check_operator_misuse
    def __eq__(self, other: str) -> "FilterExpression":
        """Create a Text equality filter expression. These expressions yield
        filters that enforce an exact match on the supplied term(s).

        Args:
            other (str): The text value to filter on.

        .. code-block:: python

            from redisvl.query.filter import Text

            f = Text("job") == "engineer"

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.EQ)
        return FilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: str) -> "FilterExpression":
        """Create a Text inequality filter expression. These expressions yield
        negated filters on exact matches on the supplied term(s). Opposite of an
        equality filter expression.

        Args:
            other (str): The text value to filter on.

        .. code-block:: python

            from redisvl.query.filter import Text

            f = Text("job") != "engineer"

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.NE)
        return FilterExpression(str(self))

    def __mod__(self, other: str) -> "FilterExpression":
        """Create a Text "LIKE" filter expression. A flexible expression that
        yields filters that can use a variety of additional operators like
        wildcards (*), fuzzy matches (%%), or combinatorics (|) of the supplied
        term(s).

        Args:
            other (str): The text value to filter on.

        .. code-block:: python

            from redisvl.query.filter import Text

            f = Text("job") % "engine*"         # suffix wild card match
            f = Text("job") % "%%engine%%"      # fuzzy match w/ Levenshtein Distance
            f = Text("job") % "engineer|doctor" # contains either term in field
            f = Text("job") % "engineer doctor" # contains both terms in field

        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, FilterOperator.LIKE)
        return FilterExpression(str(self))

    def __str__(self) -> str:
        """Return the Redis Query string for the Text filter"""
        if not self._value:
            return "*"

        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            self._value,
        )


class FilterExpression:
    """A FilterExpression is a logical combination of filters in RedisVL.

    FilterExpressions can be combined using the & and | operators to create
    complex expressions that evaluate to the Redis Query language.

    This presents an interface by which users can create complex queries
    without having to know the Redis Query language.

    .. code-block:: python

        from redisvl.query.filter import Tag, Num

        brand_is_nike = Tag("brand") == "nike"
        price_is_over_100 = Num("price") < 100
        f = brand_is_nike & price_is_over_100

        print(str(f))

        >>> (@brand:{nike} @price:[-inf (100)])

    This can be combined with the VectorQuery class to create a query:

    .. code-block:: python

        from redisvl.query import VectorQuery

        v = VectorQuery(
            vector=[0.1, 0.1, 0.5, ...],
            vector_field_name="product_embedding",
            return_fields=["product_id", "brand", "price"],
            filter_expression=f,
        )

    Note:
        Filter expressions are typically not called directly. Instead they are
        built by combining filter statements using the & and | operators.

    """

    def __init__(
        self,
        _filter: Optional[str] = None,
        operator: Optional[FilterOperator] = None,
        left: Optional["FilterExpression"] = None,
        right: Optional["FilterExpression"] = None,
    ):
        self._filter = _filter
        self._operator = operator
        self._left = left
        self._right = right

    def __and__(self, other) -> "FilterExpression":
        return FilterExpression(operator=FilterOperator.AND, left=self, right=other)

    def __or__(self, other) -> "FilterExpression":
        return FilterExpression(operator=FilterOperator.OR, left=self, right=other)

    @staticmethod
    def format_expression(left, right, operator_str) -> str:
        _left, _right = str(left), str(right)
        if _left == _right == "*":
            return _left
        if _left == "*" != _right:
            return _right
        if _right == "*" != _left:
            return _left
        return f"({_left}{operator_str}{_right})"

    def __str__(self) -> str:
        # top level check that allows recursive calls to __str__
        if not self._filter and not self._operator:
            raise ValueError("Improperly initialized FilterExpression")

        # if theres an operator, combine expressions accordingly
        if self._operator:
            if not isinstance(self._left, FilterExpression) or not isinstance(
                self._right, FilterExpression
            ):
                raise TypeError(
                    "Improper combination of filters. Both left and right should be type FilterExpression"
                )

            operator_str = " | " if self._operator == FilterOperator.OR else " "
            return self.format_expression(self._left, self._right, operator_str)

        # check that base case, the filter is set
        if not self._filter:
            raise ValueError("Improperly initialized FilterExpression")
        return self._filter


class Timestamp(Num):
    """
    A timestamp filter for querying date/time fields in Redis.

    This filter can handle various date and time formats, including:
    - datetime objects (with or without timezone)
    - date objects
    - ISO-8601 formatted strings
    - Unix timestamps (as integers or floats)

    All timestamps are converted to Unix timestamps in UTC for consistency.
    """

    SUPPORTED_TYPES = (
        datetime.datetime,
        datetime.date,
        tuple,  # Date range
        str,  # ISO format
        int,  # Unix timestamp
        float,  # Unix timestamp with fractional seconds
        type(None),
    )

    @staticmethod
    def _is_date(value: Any) -> bool:
        """Check if the value is a date object. Either ISO string or datetime.date."""
        return (
            isinstance(value, datetime.date)
            and not isinstance(value, datetime.datetime)
        ) or (isinstance(value, str) and Timestamp._is_date_only(value))

    @staticmethod
    def _is_date_only(iso_string: str) -> bool:
        """Check if an ISO formatted string only includes date information using regex."""
        # Match YYYY-MM-DD format exactly
        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        return bool(re.match(date_pattern, iso_string))

    def _convert_to_timestamp(self, value, end_date=False):
        """
        Convert various inputs to a Unix timestamp (seconds since epoch in UTC).

        Args:
            value: A datetime, date, string, int, or float

        Returns:
            float: Unix timestamp
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            # Already a Unix timestamp
            return float(value)

        if isinstance(value, str):
            # Parse ISO format
            try:
                value = datetime.datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(f"String timestamp must be in ISO format: {value}")

        if isinstance(value, datetime.date) and not isinstance(
            value, datetime.datetime
        ):
            # Convert to max or min if for dates based on end or not
            if end_date:
                value = datetime.datetime.combine(value, datetime.time.max)
            else:
                value = datetime.datetime.combine(value, datetime.time.min)

        # Ensure the datetime is timezone-aware (UTC)
        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=datetime.timezone.utc)
            else:
                value = value.astimezone(datetime.timezone.utc)

            # Convert to Unix timestamp
            return value.timestamp()

        raise TypeError(f"Unsupported type for timestamp conversion: {type(value)}")

    def __eq__(self, other) -> FilterExpression:
        """
        Filter for timestamps equal to the specified value.
        For date objects (without time), this matches the entire day.

        Args:
            other: A datetime, date, ISO string, or Unix timestamp

        Returns:
            self: The filter object for method chaining
        """
        if self._is_date(other):
            # For date objects, match the entire day
            if isinstance(other, str):
                other = datetime.datetime.strptime(other, "%Y-%m-%d").date()
            start = datetime.datetime.combine(other, datetime.time.min).astimezone(
                datetime.timezone.utc
            )
            end = datetime.datetime.combine(other, datetime.time.max).astimezone(
                datetime.timezone.utc
            )
            return self.between(start, end)

        timestamp = self._convert_to_timestamp(other)
        self._set_value(timestamp, self.SUPPORTED_TYPES, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other) -> FilterExpression:
        """
        Filter for timestamps not equal to the specified value.
        For date objects (without time), this excludes the entire day.

        Args:
            other: A datetime, date, ISO string, or Unix timestamp

        Returns:
            self: The filter object for method chaining
        """
        if self._is_date(other):
            # For date objects, exclude the entire day
            if isinstance(other, str):
                other = datetime.datetime.strptime(other, "%Y-%m-%d").date()
            start = datetime.datetime.combine(other, datetime.time.min)
            end = datetime.datetime.combine(other, datetime.time.max)
            return self.between(start, end)

        timestamp = self._convert_to_timestamp(other)
        self._set_value(timestamp, self.SUPPORTED_TYPES, FilterOperator.NE)
        return FilterExpression(str(self))

    def __gt__(self, other):
        """
        Filter for timestamps greater than the specified value.

        Args:
            other: A datetime, date, ISO string, or Unix timestamp

        Returns:
            self: The filter object for method chaining
        """
        timestamp = self._convert_to_timestamp(other)
        self._set_value(timestamp, self.SUPPORTED_TYPES, FilterOperator.GT)
        return FilterExpression(str(self))

    def __lt__(self, other):
        """
        Filter for timestamps less than the specified value.

        Args:
            other: A datetime, date, ISO string, or Unix timestamp

        Returns:
            self: The filter object for method chaining
        """
        timestamp = self._convert_to_timestamp(other)
        self._set_value(timestamp, self.SUPPORTED_TYPES, FilterOperator.LT)
        return FilterExpression(str(self))

    def __ge__(self, other):
        """
        Filter for timestamps greater than or equal to the specified value.

        Args:
            other: A datetime, date, ISO string, or Unix timestamp

        Returns:
            self: The filter object for method chaining
        """
        timestamp = self._convert_to_timestamp(other)
        self._set_value(timestamp, self.SUPPORTED_TYPES, FilterOperator.GE)
        return FilterExpression(str(self))

    def __le__(self, other):
        """
        Filter for timestamps less than or equal to the specified value.

        Args:
            other: A datetime, date, ISO string, or Unix timestamp

        Returns:
            self: The filter object for method chaining
        """
        timestamp = self._convert_to_timestamp(other)
        self._set_value(timestamp, self.SUPPORTED_TYPES, FilterOperator.LE)
        return FilterExpression(str(self))

    def between(self, start, end, inclusive: str = "both"):
        """
        Filter for timestamps between start and end (inclusive).

        Args:
            start: A datetime, date, ISO string, or Unix timestamp
            end: A datetime, date, ISO string, or Unix timestamp

        Returns:
            self: The filter object for method chaining
        """
        inclusive = self._validate_inclusive_string(inclusive)

        start_ts = self._convert_to_timestamp(start)
        end_ts = self._convert_to_timestamp(end, end_date=True)

        expression = self._format_inclusive_between(inclusive, start_ts, end_ts)

        return FilterExpression(expression)
