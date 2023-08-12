from enum import Enum
from typing import Any, List, Optional, Union

from redisvl.utils.utils import TokenEscaper


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


class FilterField:
    escaper = TokenEscaper()
    OPERATORS = []

    def __init__(self, field: str):
        self._field = field
        self._value = None
        self._operator = None

    def _set_value(self, val: Any, val_type: type, operator: FilterOperator):
        # check that the operator is supported by this class
        if operator not in self.OPERATORS:
            raise ValueError(
                f"Operator {operator} not supported by {self.__class__.__name__}. "
                + f"Supported operators are {self.OPERATORS.values()}"
            )

        if not isinstance(val, val_type):
            raise TypeError(
                f"Right side argument passed to operator {self.OPERATORS[operator]} with left side "
                f"argument {self.__class__.__name__} must be of type {val_type}"
            )
        self._value = val
        self._operator = operator


class Tag(FilterField):
    """A Tag is a FilterField representing a tag in a Redis index."""

    OPERATORS = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.IN: "==",
    }

    OPERATOR_MAP = {
        FilterOperator.EQ: "@%s:{%s}",
        FilterOperator.NE: "(-@%s:{%s})",
        FilterOperator.IN: "@%s:{%s}",
    }

    def __init__(self, field: str):
        """Create a Tag FilterField

        Args:
            field (str): The name of the tag field in the index to be queried against
        """
        super().__init__(field)

    def _set_tag_value(self, other: Union[List[str], str], operator: FilterOperator):
        if isinstance(other, list):
            if not all(isinstance(tag, str) for tag in other):
                raise ValueError("All tags must be strings")
        else:
            other = [other]
        self._set_value(other, list, operator)

    def __eq__(self, other: Union[List[str], str]) -> "FilterExpression":
        """Create a Tag equality filter expression

        Args:
            other (Union[List[str], str]): The tag(s) to filter on.

        Example:
            >>> from redisvl.query.filter import Tag
            >>> filter = Tag("brand") == "nike"
        """
        self._set_tag_value(other, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other) -> "FilterExpression":
        """Create a Tag inequality filter expression

        Args:
            other (Union[List[str], str]): The tag(s) to filter on.

        Example:
            >>> from redisvl.query.filter import Tag
            >>> filter = Tag("brand") != "nike"
        """
        self._set_tag_value(other, FilterOperator.NE)
        return FilterExpression(str(self))

    @property
    def _formatted_tag_value(self) -> str:
        return "|".join([self.escaper.escape(tag) for tag in self._value])

    def __str__(self) -> str:
        """Return the Redis Query syntax for a Tag filter expression"""
        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            self._formatted_tag_value,
        )


class Geo(FilterField):
    """A Geo is a FilterField representing a geographic (lat/lon)
    field in a Redis index.

    """
    OPERATORS = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
    }
    OPERATOR_MAP = {
        FilterOperator.EQ: "@%s:[%f %f %i %s]",
        FilterOperator.NE: "(-@%s:[%f %f %i %s])",
    }

    def __eq__(self, other) -> "FilterExpression":
        """Create a Geographic equality filter expression

        Args:
            other (GeoSpec): The geographic spec to filter on.

        Example:
            >>> from redisvl.query.filter import Geo, GeoRadius
            >>> filter = Geo("location") == GeoRadius(-122.4194, 37.7749, 1, unit="m")
        """
        self._set_value(other, GeoSpec, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other) -> "FilterExpression":
        """Create a Geographic inequality filter expression

        Args:
            other (GeoSpec): The geographic spec to filter on.

        Example:
            >>> from redisvl.query.filter import Geo, GeoRadius
            >>> filter = Geo("location") != GeoRadius(-122.4194, 37.7749, 1, unit="m")
        """
        self._set_value(other, GeoSpec, FilterOperator.NE)
        return FilterExpression(str(self))

    def __str__(self) -> "FilterExpression":
        """Return the Redis Query syntax for a Geographic filter expression"""
        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            *self._value.get_args(),
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
    """A GeoRadius is a GeoSpec representing a geographic radius"""
    def __init__(
        self,
        longitude: float,
        latitude: float,
        radius: Optional[int] = 1,
        unit: Optional[str] = "km"
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


class Num(FilterField):
    """A Num is a FilterField representing a numeric field in a Redis index."""
    OPERATORS = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.LT: "<",
        FilterOperator.GT: ">",
        FilterOperator.LE: "<=",
        FilterOperator.GE: ">=",
    }
    OPERATOR_MAP = {
        FilterOperator.EQ: "@%s:[%i %i]",
        FilterOperator.NE: "(-@%s:[%i %i])",
        FilterOperator.GT: "@%s:[(%i +inf]",
        FilterOperator.LT: "@%s:[-inf (%i]",
        FilterOperator.GE: "@%s:[%i +inf]",
        FilterOperator.LE: "@%s:[-inf %i]",
    }

    def __str__(self) -> str:
        """Return the Redis Query syntax for a Numeric filter expression"""
        if self._operator == FilterOperator.EQ or self._operator == FilterOperator.NE:
            return self.OPERATOR_MAP[self._operator] % (
                self._field,
                self._value,
                self._value,
            )
        else:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)

    def __eq__(self, other: str) -> "FilterExpression":
        """Create a Numeric equality filter expression

        Args:
            other (int): The value to filter on.

        Example:
            >>> from redisvl.query.filter import Num
            >>> filter = Num("zipcode") == 90210
        """
        self._set_value(other, int, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other: str) -> "FilterExpression":
        """Create a Numeric inequality filter expression

        Args:
            other (int): The value to filter on.

        Example:
            >>> from redisvl.query.filter import Num
            >>> filter = Num("zipcode") != 90210
        """
        self._set_value(other, int, FilterOperator.NE)
        return FilterExpression(str(self))

    def __gt__(self, other: str) -> "FilterExpression":
        """Create a Numeric greater than filter expression

        Args:
            other (int): The value to filter on.

        Example:
            >>> from redisvl.query.filter import Num
            >>> filter = Num("age") > 18
        """
        self._set_value(other, int, FilterOperator.GT)
        return FilterExpression(str(self))

    def __lt__(self, other: str) -> "FilterExpression":
        """Create a Numeric less than filter expression

        Args:
            other (int): The value to filter on.

        Example:
            >>> from redisvl.query.filter import Num
            >>> filter = Num("age") < 18
        """
        self._set_value(other, int, FilterOperator.LT)
        return FilterExpression(str(self))

    def __ge__(self, other: str) -> "FilterExpression":
        """Create a Numeric greater than or equal to filter expression

        Args:
            other (int): The value to filter on.

        Example:
            >>> from redisvl.query.filter import Num
            >>> filter = Num("age") >= 18
        """
        self._set_value(other, int, FilterOperator.GE)
        return FilterExpression(str(self))

    def __le__(self, other: str) -> "FilterExpression":
        """Create a Numeric less than or equal to filter expression

        Args:
            other (int): The value to filter on.

        Example:
            >>> from redisvl.query.filter import Num
            >>> filter = Num("age") <= 18
        """
        self._set_value(other, int, FilterOperator.LE)
        return FilterExpression(str(self))


class Text(FilterField):
    """A Text is a FilterField representing a text field in a Redis index."""
    OPERATORS = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.LIKE: "%",
    }
    OPERATOR_MAP = {
        FilterOperator.EQ: '@%s:"%s"',
        FilterOperator.NE: '(-@%s:"%s")',
        FilterOperator.LIKE: "@%s:%s",
    }

    def __eq__(self, other: str) -> "FilterExpression":
        """Create a Text equality filter expression

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from redisvl.query.filter import Text
            >>> filter = Text("job") == "engineer"
        """
        self._set_value(other, str, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other: str) -> "FilterExpression":
        """Create a Text inequality filter expression

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from redisvl.query.filter import Text
            >>> filter = Text("job") != "engineer"
        """
        self._set_value(other, str, FilterOperator.NE)
        return FilterExpression(str(self))

    def __mod__(self, other: str) -> "FilterExpression":
        """Create a Text like filter expression

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from redisvl.query.filter import Text
            >>> filter = Text("job") % "engineer"
        """
        self._set_value(other, str, FilterOperator.LIKE)
        return FilterExpression(str(self))

    def __str__(self) -> str:
        try:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)
        except KeyError:
            raise Exception("Invalid operator")


class FilterExpression:
    """A FilterExpression is a logical expression of FilterFields.

    FilterExpressions can be combined using the & and | operators to create
    complex logical expressions that evaluate to the Redis Query language.

    This presents an interface by which users can create complex queries
    without having to know the Redis Query language.

    Filter expressions are not created directly. Instead they are built
    by combining FilterFields using the & and | operators.

    Examples:

        >>> from redisvl.query.filter import Tag, Num
        >>> brand_is_nike = Tag("brand") == "nike"
        >>> price_is_over_100 = Num("price") < 100
        >>> filter = brand_is_nike & price_is_over_100
        >>> print(str(filter))
        (@brand:{nike} @price:[-inf (100)])

    This can be combined with the VectorQuery class to create a query:

        >>> from redisvl.query import VectorQuery
        >>> v = VectorQuery(
        ...     vector=[0.1, 0.1, 0.5, ...],
        ...     vector_field_name="product_embedding",
        ...     return_fields=["product_id", "brand", "price"],
        ...     filter_expression=filter,
        ... )
    """
    def __init__(
        self,
        _filter: str = None,
        operator: FilterOperator = None,
        left: Optional["FilterExpression"] = None,
        right: Optional["FilterExpression"] = None,
    ):
        self._filter = _filter
        self._operator = operator
        self._left = left
        self._right = right

    def __and__(self, other):
        return FilterExpression(operator=FilterOperator.AND, left=self, right=other)

    def __or__(self, other):
        return FilterExpression(operator=FilterOperator.OR, left=self, right=other)

    def __str__(self):
        if self._operator:
            operator_str = " | " if self._operator == FilterOperator.OR else " "
            return f"({str(self._left)}{operator_str}{str(self._right)})"
        return self._filter
